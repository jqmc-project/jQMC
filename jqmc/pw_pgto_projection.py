r"""Conversion of Quantum ESPRESSO plane-wave bands to jQMC orbital bases.

Given a Quantum ESPRESSO plane-wave reference (:mod:`jqmc.qe_wrapper`) and an
all-primitive pGTO basis seeded from a pseudopotential
(:mod:`jqmc.upf_wrapper`), this converts the Kohn--Sham bands to a jQMC
orbital basis and measures how well that basis spans them.

The reported diagnostics are:

    * V1 (fidelity): per-band residual ``r`` and the occupation-weighted mean.
    * V2 (PW reduction): basis size needed (here the all-primitive pGTO count;
      a low-G PW block can be appended for the mixed basis).
    * V3 (conditioning): condition number / smallest eigenvalue of the overlap
      matrix and the number of SVD modes dropped.

This module supports five conversions from the PW reference, all sharing the
:class:`ProjectionResult` container:

    * PW -> pGTO              (:func:`project_pw_to_pgto`)
    * PW -> pGTO + PW         (:func:`project_pw_to_mixed`)
    * PW -> PW                (:func:`project_pw_to_pw`, truncated plane waves)
    * PW -> B-spline          (:func:`project_pw_to_spline`, periodic quintic)
    * PW -> pGTO + B-spline   (:func:`project_pw_to_pgto_spline`; additive
                               composite, a single whole-cell spline)

Routes:

    * The pGTO fits (:func:`project_pw_to_pgto`, :func:`project_pw_to_mixed`
      and the pGTO step of :func:`project_pw_to_pgto_spline`) work in
      reciprocal space (:func:`build_gto_pw_matrix`). They are the routes the
      QE converters (:mod:`jqmc.qe_to_jqmc`, :mod:`jqmc.jqmc_qe`) and
      :mod:`jqmc.pgto_exponent_optimizer` call.
    * :func:`project_pw_to_pw` keeps plane waves with ``|G+k| <= g_cut``.
    * The B-spline routes (:func:`project_pw_to_spline`,
      :func:`project_pw_to_pgto_spline`) fit on the spline grid.
    * The real-space QE-grid projection (:func:`build_gto_design_matrix`) is
      the debug reference route, used by :func:`project_pw_to_pgto_debug`,
      :func:`project_pw_to_mixed_debug` and
      :func:`project_pw_to_pgto_spline_debug`. No converter calls it; the test
      suite uses it as an independent real-space cross-check.

Detailed exposition of the projection and fitting methods is deferred.

Precision Zones:
    This module owns no Precision Zone; it computes in NumPy fp64 (the FFT is fp64).

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

from __future__ import annotations

from dataclasses import dataclass, field, replace
from logging import getLogger
from math import factorial

import numpy as np
import numpy.typing as npt
import scipy.fft as _spfft

from ._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES,
    PW_PGTO_N_WORKERS,
    PW_PGTO_SVD_RCOND,
    PW_PGTO_TAIL_TOL,
    PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES,
    PW_SPLINE_FIT_CHUNK_BUDGET_BYTES,
    PW_TAPER_START_FRACTION,
    PW_TAPER_VACUUM_RADIUS_BOHR,
)
from .qe_wrapper import QEBands, QECell, QEReference, QEWavefunction
from .upf_wrapper import PrimitiveShell

logger = getLogger("jqmc").getChild(__name__)


# ---------------------------------------------------------------------------
# Fourier-space projection (PW reciprocal-lattice projection)
# ---------------------------------------------------------------------------
# Production builds the pGTO overlaps from the analytic Fourier transform of
# every Cartesian Gaussian primitive evaluated on QE's reciprocal lattice
# (:func:`build_gto_pw_matrix`), accumulating the PW-space sums over bounded
# reciprocal blocks (:func:`_project_bands_pw`). The fit needs no real-space
# FFT grid or periodic image sum. The compute is fast
# enough single-process (~ms on small fixtures, ~s on production cells) that
# it needs no MPI G-partition; ``jqmc-tool qe`` runs locally via
# ``subprocess.run``.
#
# The real-space-grid path (:func:`build_gto_design_matrix` +
# :func:`_project_bands`) exists only as a ``*_debug`` reference
# (:func:`project_pw_to_pgto_debug` etc.), used by the test suite to validate
# the Fourier production path against an independent real-space integration.


@dataclass(frozen=True)
class CartPrimitive:
    r"""A single all-primitive Cartesian Gaussian basis function.

    ``g(r) = (x-cx)^nx (y-cy)^ny (z-cz)^nz exp(-alpha |r-c|^2)`` made periodic
    by lattice translations. No contraction: each primitive exponent is its
    own basis column.
    """

    center: tuple[float, float, float]
    ang: tuple[int, int, int]
    exponent: float


def cart_primitive_normalisations(prims: list[CartPrimitive] | tuple[CartPrimitive, ...]) -> npt.NDArray[np.float64]:
    """Intrinsic Cartesian normalisations used by the emitted jQMC pGTO AOs."""
    out = np.empty(len(prims), dtype=np.float64)
    for index, prim in enumerate(prims):
        nx, ny, nz = prim.ang
        angular_l = nx + ny + nz
        angular = factorial(nx) * factorial(ny) * factorial(nz)
        angular /= factorial(2 * nx) * factorial(2 * ny) * factorial(2 * nz)
        radial = (2.0 * prim.exponent / np.pi) ** 1.5 * (8.0 * prim.exponent) ** angular_l
        out[index] = np.sqrt(radial * angular)
    return out


def periodic_pgto_overlap_matrix(
    reference: QEReference,
    prims: list[CartPrimitive] | tuple[CartPrimitive, ...],
) -> npt.NDArray[np.complex128]:
    """Raw overlap of the normalized periodic pGTO AOs emitted by the converter."""
    from ._setting import get_pgto_lattice_overlap_term_tolerance
    from .atomic_orbital import _compute_overlap_1d_cart_vectorized
    from .structure import _primitive_image_list_indices

    prims = tuple(prims)
    n_prim = len(prims)
    k_cart = np.asarray(reference.wfc.k_point, dtype=np.float64).reshape(3)
    is_gamma = not np.any(k_cart != 0.0)
    dtype = np.float64 if is_gamma else np.complex128
    overlap = np.zeros((n_prim, n_prim), dtype=dtype)
    cell = np.asarray(reference.cell.a, dtype=np.float64)
    k_frac = None if is_gamma else k_cart @ np.linalg.inv(reference.cell.b)
    tolerance = get_pgto_lattice_overlap_term_tolerance(np.float64)
    normalisations = cart_primitive_normalisations(prims)
    image_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    offsets = np.zeros((n_prim, 3), dtype=np.int64)
    different_cells = False
    if n_prim > 1:
        centers = np.asarray([prim.center for prim in prims], dtype=np.float64)
        offsets = np.floor(centers @ np.linalg.inv(cell)).astype(np.int64)
        different_cells = bool(np.any(offsets != offsets[:1]))
    for mu, prim_mu in enumerate(prims):
        center_mu = np.asarray(prim_mu.center, dtype=np.float64)
        alpha = float(prim_mu.exponent)
        for nu in range(mu, n_prim):
            prim_nu = prims[nu]
            center_nu = np.asarray(prim_nu.center, dtype=np.float64)
            beta = float(prim_nu.exponent)
            rho = alpha * beta / (alpha + beta)
            total_angular = int(sum(prim_mu.ang) + sum(prim_nu.ang))
            cache_key = (rho, total_angular)
            cached = image_cache.get(cache_key)
            if cached is None:
                indices = _primitive_image_list_indices(cell, rho, total_angular, tolerance)
                translations = np.asarray(indices, dtype=np.float64) @ cell
                cached = (indices, translations)
                image_cache[cache_key] = cached
            indices, translations = cached
            if different_cells:
                pair_offset = offsets[mu] - offsets[nu]
                if np.any(pair_offset):
                    # Keep actual image indices for the Bloch phase below.
                    indices = indices + pair_offset
                    translations = np.asarray(indices, dtype=np.float64) @ cell
            translated_nu = center_nu[None, :] + translations
            rab2 = np.sum((center_mu[None, :] - translated_nu) ** 2, axis=1)
            values = np.exp(-rho * rab2)
            zero_moment = np.sqrt(np.pi) / ((alpha + beta) ** 0.5)
            for axis in range(3):
                if prim_mu.ang[axis] == 0 and prim_nu.ang[axis] == 0:
                    values *= zero_moment
                else:
                    values *= _compute_overlap_1d_cart_vectorized(
                        alpha, beta, center_mu[axis], translated_nu[:, axis], prim_mu.ang[axis], prim_nu.ang[axis]
                    )
            if k_frac is not None:
                values = values * np.exp(2j * np.pi * (indices @ k_frac))
            value = normalisations[mu] * normalisations[nu] * np.sum(values)
            overlap[mu, nu] = value
            overlap[nu, mu] = np.conj(value)
    return np.asarray(0.5 * (overlap + overlap.conj().T), dtype=np.complex128)


@dataclass(frozen=True)
class ProjectionResult:
    """Per-band projection result and diagnostics (V1/V2/V3, see the module docstring).

    The projection is also the *conversion*: ``coefficients`` holds the pGTO
    expansion of each projected band,

        psi_n(r) ~ sum_mu coefficients[n, mu] * primitives[mu](r)
                   + sum_G pw_coefficients[n, G] * exp(i G . r),

    so ``(primitives, coefficients)`` are the pGTO part of the transform output
    and ``(pw_millers, pw_coefficients)`` the appended low-G plane-wave part
    (empty for a pGTO-only projection).

    Normalization convention: every coefficient block (pGTO, PW and
    ``spline_coefficients``) represents the unit-cell-normalized band
    ``psi_n = Omega^{-1/2} sum_G c_{n,G} e^{i(G+k).r}``, so the reconstructed
    orbital has ``integral_Omega |psi_n|^2 = 1 - r_n`` (``r_n`` the residual).
    The basis functions themselves (raw pGTO primitives, unit-amplitude plane
    waves, spline control points) are unchanged.
    """

    residuals: npt.NDArray[np.float64]  # r_n per projected band
    band_indices: npt.NDArray[np.int64]  # which bands were projected
    occupations: npt.NDArray[np.float64]  # occupation of each projected band
    coefficients: npt.NDArray[np.complex128]  # (n_band, n_gto) pGTO expansion
    primitives: tuple[CartPrimitive, ...]  # basis definition for coefficients
    n_gto: int  # all-primitive pGTO columns
    n_pw: int  # low-G PW columns (0 if pGTO-only)
    n_dropped: int  # SVD modes removed (lin. dependency)
    s_cond: float  # cond(S) after truncation
    s_cond_full: float  # cond(S) before truncation (V3)
    s_eig_min: float  # smallest eigenvalue of S
    pw_millers: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )  # (n_pw, 3) retained PW Miller indices
    pw_coefficients: npt.NDArray[np.complex128] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.complex128)
    )  # (n_band, n_pw) PW expansion
    alpha_min: float = 0.0  # diffuse-pGTO floor applied (0 = none)
    g_cut: float = 0.0  # PW handoff cutoff |G+k| (0 = pGTO-only)
    spline_coefficients: npt.NDArray[np.complex128] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.complex128)
    )  # (n_band, prod(spline_shape)) B-spline coeffs
    spline_shape: tuple[int, int, int] | tuple[()] = ()  # B-spline grid (0,0,0 if none)
    spline_order: int = 0  # B-spline degree (3 cubic / 5 quintic)
    mesh_factor: float = 0.0  # spline grid spacing ratio (0 = no spline)

    @property
    def mean_residual(self) -> float:
        """Occupation-weighted mean residual (V1 headline number)."""
        w = self.occupations
        if w.sum() <= 0:
            return float(np.mean(self.residuals))
        return float(np.sum(w * self.residuals) / np.sum(w))

    @property
    def max_residual(self) -> float:
        return float(np.max(self.residuals)) if self.residuals.size else float("nan")


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------
def pgto_leaf_image_tolerances(
    prims: list[CartPrimitive] | tuple[CartPrimitive, ...], tol: float = AO_IMAGE_TRUNCATION_ATOL
) -> npt.NDArray[np.float64]:
    r"""Per-primitive envelope tolerance of the emitted jQMC pGTO leaf, for the real-space image sums.

    One tolerance is returned per emitted primitive. Host-side fp64.
    """
    from .atomic_orbital import _primitive_value_envelope_prefactor

    cache: dict[tuple[float, int], float] = {}
    out = np.empty(len(prims), dtype=np.float64)
    for index, prim in enumerate(prims):
        key = (float(prim.exponent), int(sum(prim.ang)))
        if key not in cache:
            cache[key] = float(tol) / max(1.0, _primitive_value_envelope_prefactor(*key))
        out[index] = cache[key]
    return out


def build_pgto_primitives(
    shells_per_atom: list[list[PrimitiveShell]],
    atom_positions: npt.NDArray[np.float64],
) -> list[CartPrimitive]:
    r"""Expand per-atom fitted shells into an all-primitive Cartesian pGTO set.

    For every atom, every shell ``l``, every primitive exponent in that shell
    and every Cartesian component of ``l`` one basis function is emitted (unit
    coefficient). The shell fit coefficients are not used in the emitted basis.

    Args:
        shells_per_atom: ``shells_per_atom[i]`` are the shells for atom ``i``.
        atom_positions: ``(n_atom, 3)`` Cartesian centres (Bohr).

    Returns:
        Flat list of :class:`CartPrimitive`.
    """
    if len(shells_per_atom) != len(atom_positions):
        raise ValueError("shells_per_atom and atom_positions length mismatch.")
    prims: list[CartPrimitive] = []
    for shells, pos in zip(shells_per_atom, atom_positions, strict=True):
        center = (float(pos[0]), float(pos[1]), float(pos[2]))
        for shell in shells:
            angular_l = shell.angular_l
            # Cartesian components (nx, ny, nz), nx + ny + nz == l, in canonical alphabetic
            # order (nx descending, then ny descending): l=2 -> xx, xy, xz, yy, yz, zz.
            for ang in [
                (nx, ny, angular_l - nx - ny) for nx in range(angular_l, -1, -1) for ny in range(angular_l - nx, -1, -1)
            ]:
                for alpha in shell.exponents:
                    prims.append(CartPrimitive(center=center, ang=ang, exponent=float(alpha)))
    return prims


def alpha_min_from_gcut(g_cut: float, tail_tol: float = PW_PGTO_TAIL_TOL) -> float:
    r"""Diffuse-pGTO floor matched to a plane-wave handoff cutoff ``g_cut``.

    In the mixed pGTO + PW basis the plane-wave block carries completeness, so
    primitives that the retained PW shell already represents are redundant and
    only worsen the overlap conditioning. Primitives with ``exponent <
    alpha_min`` are dropped, where ``alpha_min`` is the ``g_cut``-matched floor.
    ``g_cut`` is in Bohr^-1. ``g_cut <= 0`` disables the floor (returns 0,
    keeping every primitive).

    ``tail_tol`` is the dimensionless Gaussian momentum-envelope tolerance
    used to choose this floor. It defaults to
    :data:`jqmc._setting.PW_PGTO_TAIL_TOL` (currently ``1e-3``).
    """
    if g_cut <= 0.0:
        return 0.0
    return float(g_cut**2 / (4.0 * max(-np.log(tail_tol), 1.0)))


def filter_shells_by_alpha_min(shells_per_atom: list[list[PrimitiveShell]], alpha_min: float) -> list[list[PrimitiveShell]]:
    """Drop primitives with ``exponent < alpha_min`` (diffuse handoff to PW).

    Shells left with no surviving primitive are removed. The fit diagnostics
    (``rel_error``/``overlap_cond``) are carried over unchanged --- they only
    described the original fit and are not used downstream.
    """
    if alpha_min <= 0.0:
        return [list(shells) for shells in shells_per_atom]
    out: list[list[PrimitiveShell]] = []
    for shells in shells_per_atom:
        kept_shells: list[PrimitiveShell] = []
        for sh in shells:
            kept = [(a, c) for a, c in zip(sh.exponents, sh.coefficients, strict=True) if a >= alpha_min]
            if not kept:
                continue
            exps, coeffs = zip(*kept, strict=True)
            kept_shells.append(
                PrimitiveShell(
                    angular_l=sh.angular_l,
                    exponents=tuple(float(a) for a in exps),
                    coefficients=tuple(float(c) for c in coeffs),
                    rel_error=sh.rel_error,
                    overlap_cond=sh.overlap_cond,
                )
            )
        out.append(kept_shells)
    return out


# ---------------------------------------------------------------------------
# Real-space grid sampling
# ---------------------------------------------------------------------------
def grid_points(
    cell: QECell,
    fft_grid: tuple[int, int, int],
    shift_frac: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> npt.NDArray[np.float64]:
    """Cartesian coordinates of the dense FFT mesh, shape ``(n_points, 3)``.

    ``shift_frac`` (in units of the grid step) offsets every point, e.g.
    ``(0.5, 0.5, 0.5)`` samples the cell midpoints --- used to evaluate the pGTO
    part on the same half-shifted mesh as the spline fidelity check.
    """
    n1, n2, n3 = fft_grid
    f1 = (np.arange(n1) + shift_frac[0]) / n1
    f2 = (np.arange(n2) + shift_frac[1]) / n2
    f3 = (np.arange(n3) + shift_frac[2]) / n3
    frac = np.stack(np.meshgrid(f1, f2, f3, indexing="ij"), axis=-1).reshape(-1, 3)
    return frac @ cell.a


def fft_cube_millers(fft_grid: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    r"""Signed Miller indices of every G in the dense FFT cube, shape ``(N, 3)``.

    Each axis uses the standard FFT-shifted convention: integer indices
    ``[0, n/2]`` map to positive G's and ``(n/2, n)`` map to negative G's
    (``m - n``). ``N = prod(fft_grid)`` and the order matches
    ``np.meshgrid(..., indexing='ij')`` of those per-axis ranges.
    """

    def axis_signed(n: int) -> npt.NDArray[np.int64]:
        pos = np.arange((n + 1) // 2, dtype=np.int64)
        neg = np.arange(-(n // 2), 0, dtype=np.int64)
        return np.concatenate([pos, neg])

    m1 = axis_signed(int(fft_grid[0]))
    m2 = axis_signed(int(fft_grid[1]))
    m3 = axis_signed(int(fft_grid[2]))
    g1, g2, g3 = np.meshgrid(m1, m2, m3, indexing="ij")
    return np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)


def _canonical_twist_cart(reference: QEReference) -> npt.NDArray[np.float64]:
    """Return the Cartesian twist in the canonical ``[-1/2, 1/2)`` BZ.

    A reciprocal-gauge transport ``k' = k + s``, ``m' = m - s`` leaves every
    physical wavevector ``G+k`` unchanged.  The finite FFT Miller cube used
    for the mixed pGTO Gram matrix has no transported source rows, however, so
    evaluating that cube with the raw ``k'`` would make the self-overlap depend
    on the arbitrary reciprocal label.  Reducing the fractional twist before
    that auxiliary sum gives every equivalent source the same finite cube.

    Twists already in the canonical zone are returned without a round trip so
    existing Gamma and ordinary QE conversions remain byte-for-byte stable.
    """
    k_cart = np.asarray(reference.wfc.k_point, dtype=np.float64).reshape(3)
    k_frac = np.linalg.solve(reference.cell.b.T, k_cart)
    reciprocal_shift = np.floor(k_frac + 0.5).astype(np.int64)
    if not np.any(reciprocal_shift):
        return k_cart.copy()
    return (k_frac - reciprocal_shift.astype(np.float64)) @ reference.cell.b


def _envelope_radius(min_exp: float, ang_l: int, tol: float) -> float:
    """Radius where ``r^l exp(-min_exp r^2)`` drops below ``tol`` of its peak.

    Solves the implicit equation ``log_inv_tol = min_exp * r^2 - l * log(r)``
    for ``r`` by a few Picard iterations (the ``l log r`` term is sub-leading
    and converges in 3-5 sweeps). For ``l = 0`` the closed form
    ``sqrt(log_inv_tol / min_exp)`` is exact.
    """
    log_inv_tol = max(-float(np.log(tol)), 1.0)
    if ang_l <= 0:
        return float(np.sqrt(log_inv_tol / min_exp))
    r_cut = float(np.sqrt(log_inv_tol / min_exp))
    for _ in range(5):
        r_cut = float(np.sqrt((log_inv_tol + ang_l * np.log(max(r_cut, 1.0))) / min_exp))
    return r_cut


def _image_translations(
    cell: QECell, min_exp: float, ang_l: int, tol: float = AO_IMAGE_TRUNCATION_ATOL, *, center=None
) -> npt.NDArray[np.float64]:
    """Lattice translations whose Gaussian tails reach the cell interior.

    The reach radius is the per-primitive envelope :func:`_envelope_radius`
    plus the cell body diagonal so a Gaussian centred anywhere inside the
    cell is covered. For an off-cell centre, translate the integer image box
    back by its cell offset, retaining actual translations for Bloch phases.
    """
    r_cut = _envelope_radius(min_exp, ang_l, tol) + float(np.linalg.norm(cell.a.sum(axis=0)))
    reciprocal = cell.b
    b_norms = np.linalg.norm(reciprocal, axis=1)
    plane_spacing = (2.0 * np.pi) / b_norms
    n_max = (np.ceil(r_cut / plane_spacing).astype(int) + 1).tolist()
    rngs = [range(-n, n + 1) for n in n_max]
    tvecs = np.stack(np.meshgrid(*rngs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    if center is not None:
        offset = np.floor(np.asarray(center) @ reciprocal.T / (2.0 * np.pi))
        if np.any(offset):
            tvecs -= offset
    return tvecs @ cell.a


def eval_primitive_on_grid(
    cell: QECell,
    prim: CartPrimitive,
    grid: npt.NDArray[np.float64],
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> npt.NDArray[np.float64]:
    """Evaluate one Gamma-point periodic Cartesian primitive on ``grid``."""
    center = np.asarray(prim.center, dtype=np.float64)
    nx, ny, nz = prim.ang
    alpha = prim.exponent
    tvecs = _image_translations(cell, alpha, nx + ny + nz, tol, center=center)
    out = np.zeros(grid.shape[0], dtype=np.float64)
    for t in tvecs:
        d = grid - (center + t)
        r2 = np.einsum("ij,ij->i", d, d)
        poly = (d[:, 0] ** nx) * (d[:, 1] ** ny) * (d[:, 2] ** nz)
        out += poly * np.exp(-alpha * r2)
    return out


def _eval_design_matrix_group(
    g_idx: int,
    center_t: tuple[float, float, float],
    alpha: float,
    cols: list[int],
    cols_ang: list[tuple[int, int, int]],
    cell: QECell,
    grid: npt.NDArray[np.float64],
    cell_min: npt.NDArray[np.float64],
    cell_max: npt.NDArray[np.float64],
    tol: float,
    k_cart: npt.NDArray[np.float64] | None = None,
) -> tuple[int, list[int], np.ndarray | None, int, int, float, int, float]:
    """Evaluate one ``(center, alpha)`` group of the GTO design matrix.

    ``k_cart`` (Cartesian ``1/bohr``): when given, each image contribution
    is weighted by the Bloch phase ``e^{i k . T}`` (``T`` = image
    translation), turning the plain periodic sum into the Bloch image sum
    ``sum_T g(r - c - T) e^{i k . T}`` -- the same function the jQMC
    complex pGTO leaf evaluates inside the home cell. Output dtype becomes
    ``complex128``. ``None`` keeps the real Gamma path byte-identical.

    Pure function -- reads no shared mutable state, writes nothing
    outside its returned local array. Safe to call concurrently across
    groups because each group owns a disjoint slab of output columns
    (``cols``). Used by :func:`build_gto_design_matrix` driving via
    ``joblib.Parallel(prefer="threads")``; the GIL is released inside
    the dominant kernel (``np.exp`` + polynomial accumulation) so
    threads scale on multi-core CPUs to the **memory-bandwidth limit**
    (~3 effective workers on a laptop for 1.55M-point grids;
    additional workers do not help because element-wise ``np.exp`` on a
    12 MB array is bound by DRAM throughput).

    Inner loop has two fast paths to reduce wasted memory traffic
    (which dominates per-anchor cost on bandwidth-limited hardware):

    * **s-orbital** (``(nx, ny, nz) == (0, 0, 0)``): skip the
      ``(dx**0)(dy**0)(dz**0)`` chain entirely -- those would allocate
      three ones-arrays and two multiplies, doubling memory traffic for
      no work. Accumulate ``radial`` directly.
    * **single-axis** (one of nx/ny/nz > 0, others 0): avoid
      ``dx**0``-style allocations and ``dx**1``-style copies; reuse the
      relevant ``dx`` / ``dy`` / ``dz`` view directly.

    Returns ``(g_idx, cols, group_out, n_images_kept, n_images_total,
    alpha, l_max, t_group_seconds)``. ``group_out`` is ``None`` when no
    images survive screening (the caller skips the write in that case).
    """
    import time

    t0 = time.monotonic()
    center = np.asarray(center_t, dtype=np.float64)
    l_max = max(sum(a) for a in cols_ang)
    tvecs = _image_translations(cell, alpha, l_max, tol, center=center)
    # Per-image envelope half-width (Cartesian). Same r_cut formula as
    # _image_translations, but without the cell-diagonal padding that
    # only matters for image inclusion.
    r_cut = _envelope_radius(alpha, l_max, tol)
    # Pre-screen tvecs: an image whose anchor bbox does not overlap the
    # cell bbox cannot contribute, regardless of grid resolution.
    anchors = center[None, :] + tvecs
    keep = np.all(anchors > cell_min[None, :] - r_cut, axis=1) & np.all(anchors < cell_max[None, :] + r_cut, axis=1)
    tvecs_kept = tvecs[keep]
    n_images_kept = int(tvecs_kept.shape[0])
    n_images_total = int(tvecs.shape[0])

    n_points = grid.shape[0]
    if n_images_kept == 0:
        return g_idx, cols, None, n_images_kept, n_images_total, alpha, l_max, time.monotonic() - t0

    gx = grid[:, 0]
    gy = grid[:, 1]
    gz = grid[:, 2]
    anchors_kept = center[None, :] + tvecs_kept  # (n_kept, 3)
    # Bloch image phases e^{i k . T} (complex path); None on the Gamma path.
    if k_cart is not None:
        image_phases = np.exp(1j * (tvecs_kept @ np.asarray(k_cart, dtype=np.float64)))
        group_out = np.zeros((n_points, len(cols)), dtype=np.complex128)
    else:
        image_phases = None
        group_out = np.zeros((n_points, len(cols)), dtype=np.float64)
    for i_img, anchor in enumerate(anchors_kept):
        dx = gx - anchor[0]
        dy = gy - anchor[1]
        dz = gz - anchor[2]
        radial = np.exp(-alpha * (dx * dx + dy * dy + dz * dz))  # (n_points,)
        if image_phases is not None:
            radial = radial * image_phases[i_img]  # complex from here on
        for c, (nx, ny, nz) in enumerate(cols_ang):
            # Fast paths: skip wasted ** allocations for low-l components.
            if nx == 0 and ny == 0 and nz == 0:
                group_out[:, c] += radial
            else:
                # Build poly = dx^nx * dy^ny * dz^nz, skipping any factor
                # whose exponent is 0 (would alloc a ones-array) and using
                # the ndarray view directly when the exponent is 1 (would
                # otherwise alloc a copy via ``**1``).
                poly = None
                if nx > 0:
                    poly = dx if nx == 1 else dx**nx
                if ny > 0:
                    term = dy if ny == 1 else dy**ny
                    poly = term if poly is None else poly * term
                if nz > 0:
                    term = dz if nz == 1 else dz**nz
                    poly = term if poly is None else poly * term
                group_out[:, c] += poly * radial
    return g_idx, cols, group_out, n_images_kept, n_images_total, alpha, l_max, time.monotonic() - t0


def build_gto_design_matrix(
    cell: QECell,
    prims: list[CartPrimitive],
    grid: npt.NDArray[np.float64],
    tol: float | npt.ArrayLike = AO_IMAGE_TRUNCATION_ATOL,
    n_jobs: int | None = None,
    k_cart: npt.NDArray[np.float64] | tuple[float, float, float] | None = None,
) -> npt.NDArray[np.complex128]:
    r"""Sample all primitives on the grid; returns ``(n_points, n_gto)`` real.

    ``k_cart`` (Cartesian ``1/bohr``): when given and non-zero, each image
    contribution carries the Bloch phase ``e^{i k . T}`` so the columns are
    the Bloch image sums; the output is then ``complex128``. ``None`` / zero
    keeps the real Gamma path.

    Real-space reference path used by the ``*_debug`` projectors only;
    production uses the Fourier projection. Pure numpy (no JAX).

    ``n_jobs`` defaults to :data:`jqmc._setting.PW_PGTO_N_WORKERS`
    (``-1`` = all cores). Pass ``n_jobs=1`` to bypass joblib entirely
    (identical numerics, no thread-pool overhead).

    ``tol`` is one envelope tolerance for every primitive, or a
    ``(len(prims),)`` per-primitive array (e.g.
    :func:`pgto_leaf_image_tolerances`); a ``(center, exponent)`` group then
    uses the smallest tolerance of its columns.
    """
    n_points = grid.shape[0]
    tol_arr = np.broadcast_to(np.asarray(tol, dtype=np.float64), (len(prims),))
    k_cart_arr: npt.NDArray[np.float64] | None
    if k_cart is None:
        k_cart_arr = None
    else:
        k_cart_arr = np.asarray(k_cart, dtype=np.float64).reshape(3)
        if not np.any(k_cart_arr != 0.0):
            k_cart_arr = None  # Gamma: keep the real fast path.
    out = np.zeros((n_points, len(prims)), dtype=np.float64 if k_cart_arr is None else np.complex128)
    # Axis-aligned bbox of the cell parallelepiped (8 corners). For an
    # orthogonal cell this is tight; for non-orthogonal it is conservative.
    _corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64) @ cell.a
    cell_min = _corners.min(axis=0)
    cell_max = _corners.max(axis=0)

    # Group column indices by (center, exponent), keeping the max l for the
    # image radius so every component in the group is covered.
    groups: dict[tuple[tuple[float, float, float], float], list[int]] = {}
    for j, p in enumerate(prims):
        groups.setdefault((p.center, p.exponent), []).append(j)

    # Process diffuse groups first so the dominant cost is paid up front
    # (tight Gaussians have ~1 image and finish instantly; diffuse Gaussians
    # carry the runtime).
    group_items = sorted(groups.items(), key=lambda kv: kv[0][1])

    # Progress reporting: large FFT grids + many diffuse primitives can
    # take minutes; emit one INFO line per group and a final summary so
    # the user sees the loop is alive without needing tqdm.
    import time

    n_groups = len(group_items)
    log_every = max(1, n_groups // 20)  # ~20 progress lines max
    t_start = time.monotonic()
    resolved_n_jobs = PW_PGTO_N_WORKERS if n_jobs is None else int(n_jobs)
    logger.info(
        "build_gto_design_matrix: %d primitives in %d (center, alpha) groups on %d grid points; "
        "alpha range [%.4g, %.4g]; n_jobs=%d.",
        len(prims),
        n_groups,
        n_points,
        group_items[0][0][1] if group_items else 0.0,
        group_items[-1][0][1] if group_items else 0.0,
        resolved_n_jobs,
    )

    # Pre-pack per-group inputs (the helper is pure; everything it needs
    # comes via arguments).
    group_args = [
        (
            g_idx,
            center_t,
            alpha,
            cols,
            [prims[j].ang for j in cols],
            float(np.min(tol_arr[cols])),
        )
        for g_idx, ((center_t, alpha), cols) in enumerate(group_items)
    ]

    if resolved_n_jobs == 1:
        # Serial fast path: skip joblib's worker / dispatch overhead
        # for the small-group case (and for use under nested parallel
        # contexts that already saturate the CPU).
        results = [
            _eval_design_matrix_group(g_idx, ct, a, cls, cls_ang, cell, grid, cell_min, cell_max, g_tol, k_cart=k_cart_arr)
            for (g_idx, ct, a, cls, cls_ang, g_tol) in group_args
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=resolved_n_jobs, prefer="threads")(
            delayed(_eval_design_matrix_group)(
                g_idx, ct, a, cls, cls_ang, cell, grid, cell_min, cell_max, g_tol, k_cart=k_cart_arr
            )
            for (g_idx, ct, a, cls, cls_ang, g_tol) in group_args
        )

    # Aggregate: completion order under threads is non-deterministic,
    # so sort by g_idx to keep the progress log lines monotonic for the
    # user. Each group writes to disjoint output columns so the write
    # order does not affect ``out`` numerically.
    results.sort(key=lambda r: r[0])
    total_images = 0
    for g_idx, cols, group_out, n_images_kept, n_images_total, alpha, l_max, t_group in results:
        total_images += n_images_kept
        if group_out is not None:
            out[:, cols] = group_out
        if ((g_idx + 1) % log_every == 0) or (n_images_kept > 100):
            logger.info(
                "  group %4d/%-4d done in %6.1fs: alpha=%.4g l_max=%d n_images_kept=%d/%d cols=%d.",
                g_idx + 1,
                n_groups,
                t_group,
                alpha,
                l_max,
                n_images_kept,
                n_images_total,
                len(cols),
            )

    logger.info(
        "build_gto_design_matrix: done in %.1fs, total kept images=%d.",
        time.monotonic() - t_start,
        total_images,
    )
    # Return float64. Downstream consumers (_project_bands,
    # evaluate_pgto_expansion, _gto_fourier_at_millers) all accept a
    # real array: ``phi.conj()`` is a no-op on real arrays, and
    # ``phi.T @ phi`` / ``phi.T @ psi_complex`` produce the correct
    # real-symmetric / complex result via numpy broadcasting. Cast to
    # complex inside any consumer that genuinely needs a complex
    # working array (e.g. _gto_fourier_at_millers' FFT path).
    return out


def evaluate_pgto_expansion(
    cell: QECell,
    primitives: tuple[CartPrimitive, ...] | list[CartPrimitive],
    coefficients: npt.NDArray[np.complex128],
    grid: npt.NDArray[np.float64],
    tol: float | npt.ArrayLike = AO_IMAGE_TRUNCATION_ATOL,
) -> npt.NDArray[np.complex128]:
    """Evaluate pGTO-expanded orbitals on ``grid`` from conversion coefficients.

    Inverse of the projection: given ``(primitives, coefficients)`` from a
    :class:`ProjectionResult`, returns ``(n_points, n_band)`` with column ``n``
    equal to ``sum_mu coefficients[n, mu] * primitives[mu](r)``. Useful to
    verify the conversion or to resample the pGTO orbitals on any grid.
    """
    phi = build_gto_design_matrix(cell, list(primitives), grid, tol=tol)
    return phi @ np.asarray(coefficients, dtype=np.complex128).T


def _u_gauge_design_matrix(
    cell: QECell,
    prims: list[CartPrimitive],
    grid: npt.NDArray[np.float64],
    k_cart: npt.ArrayLike,
    tol: float | npt.ArrayLike,
) -> npt.NDArray:
    r"""Real-space pGTO design matrix in the cell-periodic u-gauge.

    Column ``mu`` is ``e^{-i k . r} sum_T g_mu(r - c_mu - T) e^{i k . T}`` --
    the periodic part of the Bloch image sum the jQMC complex pGTO leaf
    evaluates. Fitting against ``reconstruct_pw_on_grid`` output (also the
    periodic part) in this gauge yields the same coefficients as the
    Fourier fit of the full Bloch orbital at ``q = G + k``. Gamma (zero
    ``k_cart``) reduces to the plain real :func:`build_gto_design_matrix`.
    """
    k_arr = np.asarray(k_cart, dtype=np.float64).reshape(3)
    if not np.any(k_arr != 0.0):
        return build_gto_design_matrix(cell, prims, grid, tol=tol)
    phi = build_gto_design_matrix(cell, prims, grid, tol=tol, k_cart=k_arr)
    return phi * np.exp(-1j * (np.asarray(grid, dtype=np.float64) @ k_arr))[:, None]


def evaluate_pgto_expansion_u_gauge(
    cell: QECell,
    primitives: tuple[CartPrimitive, ...] | list[CartPrimitive],
    coefficients: npt.NDArray[np.complex128],
    grid: npt.NDArray[np.float64],
    k_cart: npt.NDArray[np.float64] | tuple[float, float, float] | None,
    tol: float | npt.ArrayLike = AO_IMAGE_TRUNCATION_ATOL,
) -> npt.NDArray[np.complex128]:
    r"""Periodic (u-gauge) part of the Bloch pGTO expansion on ``grid``.

    Returns the cell-periodic part of the Bloch image sum that the jQMC
    complex pGTO leaf evaluates. This is the correct object to subtract from
    :func:`reconstruct_pw_on_grid` output (which is also the periodic part)
    when forming a real-space residual at a non-Gamma twist: the plain
    :func:`evaluate_pgto_expansion` misses both the image phases and the
    envelope, which corrupts the residual by O(1) at general k.

    ``k_cart`` of ``None`` / zeros reduces byte-identically to
    :func:`evaluate_pgto_expansion` (Gamma path).
    """
    if k_cart is None:
        return evaluate_pgto_expansion(cell, primitives, coefficients, grid, tol=tol)
    k_arr = np.asarray(k_cart, dtype=np.float64).reshape(3)
    if not np.any(k_arr != 0.0):
        return evaluate_pgto_expansion(cell, primitives, coefficients, grid, tol=tol)
    phi = build_gto_design_matrix(cell, list(primitives), grid, tol=tol, k_cart=k_arr)
    vals = phi @ np.asarray(coefficients, dtype=np.complex128).T  # (n_points, n_band)
    envelope = np.exp(-1j * (np.asarray(grid, dtype=np.float64) @ k_arr))  # e^{-i k . r}
    return vals * envelope[:, None]


def reconstruct_pw_orbitals(wfc: QEWavefunction, fft_grid: tuple[int, int, int]) -> npt.NDArray[np.complex128]:
    r"""Reconstruct PW bands on the real-space FFT mesh, shape ``(n_points, n_bnd)``.

    ``psi(r) = sum_G c_G exp(iG.r)`` is evaluated on the mesh whose index
    ``(i,j,k)`` maps to fractional ``(i/n1, j/n2, k/n3)``.
    """
    return reconstruct_pw_on_grid(wfc, fft_grid)


def reconstruct_pw_on_grid(
    wfc: QEWavefunction,
    grid_shape: tuple[int, int, int],
    shift_frac: tuple[float, float, float] | None = None,
    band_subset: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.complex128]:
    r"""Reconstruct PW bands on a uniform mesh ``grid_shape``, ``(n_points, n_bnd)``.

    ``psi(r) = sum_G c_G exp(iG.r)`` is evaluated on the uniform grid whose point
    ``(i,j,k)`` is fractional ``((i+sx)/m1, (j+sy)/m2, (k+sz)/m3)``. A non-zero
    ``shift_frac`` ``(sx,sy,sz)`` (in units of the grid step) shifts the sampled
    mesh. Raises on aliasing (Miller index beyond Nyquist).

    ``band_subset`` selects band rows before the transform, so only those
    bands are materialised on the grid (each band's transform is independent,
    so the result is bit-identical to slicing the full output with
    ``[:, band_subset]``).
    """
    m = np.array(grid_shape, dtype=int)
    millers = wfc.millers
    if np.any(np.abs(millers) > m // 2):
        raise ValueError("Miller indices exceed the grid Nyquist range (aliasing).")
    idx = (millers % m).astype(int)
    coeffs = wfc.coeffs if band_subset is None else wfc.coeffs[np.asarray(band_subset, dtype=np.int64)]
    if shift_frac is not None and any(s != 0.0 for s in shift_frac):
        phase = np.exp(2j * np.pi * (millers.astype(np.float64) @ (np.asarray(shift_frac) / m)))
        coeffs = coeffs * phase[None, :]
    # Batched scatter + single 3D IFFT over the selected bands.
    # Memory: n_sel * prod(m) * 16 B.
    n_sel = coeffs.shape[0]
    npts = int(m[0] * m[1] * m[2])
    cube = np.zeros((n_sel, m[0], m[1], m[2]), dtype=np.complex128)
    cube[:, idx[:, 0], idx[:, 1], idx[:, 2]] = coeffs
    # NOTE: deliberately NOT ``overwrite_x=True`` here. It would halve this
    # function's peak, but pocketfft's in-place multi-axis ``ifftn`` picks a
    # different pass order and the result is NOT bit-identical (measured
    # rel 3.4e-16 on the H-ae QE fixture, amplified to ~1e-14 through the
    # prefilter's deconvolution). These coefficients ARE the wavefunction every
    # downstream run consumes, so exactness wins; the peak is bounded instead
    # by the band chunking in ``fit_pw_to_spline``, which is exact by
    # construction. (The per-axis 1D chain in ``bspline_prefilter_periodic``
    # WAS verified bit-identical in place, and does use ``overwrite_x``.)
    real = _spfft.ifftn(cube, axes=(1, 2, 3), workers=PW_PGTO_N_WORKERS) * npts
    return real.reshape(n_sel, npts).T


# ---------------------------------------------------------------------------
# G-sphere taper of the plane-wave coefficients (vacuum cells only)
# ---------------------------------------------------------------------------
# A sphere-truncated expansion psi = sum_{|G+k| <= G_max} c_G e^{i(G+k).r} is a
# finite Fourier sum, entire in r, but its truncation remainder does not decay
# with the distance from the nuclei: far from every nucleus the exponential
# tail of the true orbital falls below a truncation-ringing (Gibbs) floor whose
# amplitude is set by the coefficients at the sphere edge and which oscillates
# with wavenumber ~G_max, where -1/2 lap(psi)/psi is O(+-G_max^2/2) of either
# sign. Multiplying the coefficients by a C-infinity window that reaches zero at
# G_max with all derivatives removes the edge: the remainder of the windowed sum
# decays super-algebraically (not exponentially) with the distance, so the local
# kinetic energy is restored wherever the physical tail still exceeds that
# remainder, and no local-energy bound holds beyond it. See PW_TAPER_* in
# jqmc/_setting.py for the incident, the numbers and why a lower hard cutoff
# does not help.

#: Conversion kinds whose Hamiltonian carries plane-wave coefficients (directly
#: or through a B-spline fitted to them) and are therefore tapered.
PW_TAPER_KINDS: tuple[str, ...] = ("pw", "bspline", "pgto+pw", "pgto+bspline")


def pw_taper_window(q: npt.ArrayLike, g_start: float, g_end: float) -> npt.NDArray[np.float64]:
    r"""C-infinity smooth step in ``|G+k|``: 1 for ``q <= g_start``, 0 for ``q >= g_end``.

    The transition is monotone and has vanishing derivatives at both ends, so
    the tapered spectrum is smooth at the sphere edge. Detailed window formula
    is work in progress.
    """
    if not g_end > g_start > 0.0:
        raise ValueError(f"pw_taper_window needs 0 < g_start < g_end, got g_start={g_start}, g_end={g_end}.")
    qv = np.asarray(q, dtype=np.float64)
    t = np.clip((g_end - qv) / (g_end - g_start), 0.0, 1.0)  # 1 at g_start, 0 at g_end
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        f_t = np.where(t > 0.0, np.exp(-1.0 / np.where(t > 0.0, t, 1.0)), 0.0)
        f_u = np.where(t < 1.0, np.exp(-1.0 / np.where(t < 1.0, 1.0 - t, 1.0)), 0.0)
    w = f_t / (f_t + f_u)
    w[t >= 1.0] = 1.0
    w[t <= 0.0] = 0.0
    return w


def max_empty_distance(cell_a: npt.NDArray[np.float64], atom_positions: npt.NDArray[np.float64]) -> float:
    r"""Covering radius (bohr) of the periodic nuclei: the largest distance from any point to its nearest nucleus.

    The result is exact up to floating point and invariant under the choice of
    origin, of the primitive basis and of the lattice-translation
    representatives of the atoms. Detailed geometric construction is work in
    progress.
    """
    from scipy.spatial import Voronoi

    a = np.asarray(cell_a, dtype=np.float64)
    pos = np.asarray(atom_positions, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] == 0:
        raise ValueError("max_empty_distance needs at least one atom.")
    frac = pos @ np.linalg.inv(a)
    frac -= np.floor(frac)  # canonical representatives in [0, 1)
    volume = abs(float(np.linalg.det(a)))
    cross = np.stack([np.cross(a[1], a[2]), np.cross(a[2], a[0]), np.cross(a[0], a[1])])
    heights = volume / np.linalg.norm(cross, axis=1)
    # No empty sphere is larger than half the longest body diagonal of the cell,
    # so every generator of a Voronoi vertex inside the cell lies within that
    # distance of the cell; one extra shell covers the cell's own extent.
    signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], dtype=np.float64)
    r_bound = 0.5 * float(np.max(np.linalg.norm(signs @ a, axis=1)))
    n_img = [int(np.ceil(r_bound / h)) + 1 for h in heights]
    shifts = np.stack(np.meshgrid(*[np.arange(-n, n + 1) for n in n_img], indexing="ij"), axis=-1).reshape(-1, 3)
    generators = ((frac[:, None, :] + shifts[None, :, :].astype(np.float64)) @ a).reshape(-1, 3)
    vertices = Voronoi(generators).vertices
    v_frac = vertices @ np.linalg.inv(a)
    inside = np.all((v_frac >= -1.0e-9) & (v_frac < 1.0 + 1.0e-9), axis=1) & np.all(np.isfinite(vertices), axis=1)
    inside_vertices = vertices[inside]
    radii = np.empty(inside_vertices.shape[0])
    for i, v in enumerate(inside_vertices):
        radii[i] = float(np.min(np.linalg.norm(generators - v[None, :], axis=1)))
    return float(np.max(radii))


def taper_qe_reference(
    reference: QEReference,
    *,
    g_end: float | None = None,
    label: str = "",
) -> tuple[QEReference, dict[str, float | npt.NDArray[np.float64]] | None]:
    r"""Taper the PW coefficients at the G-sphere edge when the cell has a vacuum region.

    Returns ``(reference, None)`` -- the very same object, so the conversion is
    bit-identical -- when no point of the cell is farther than
    ``PW_TAPER_VACUUM_RADIUS_BOHR`` from every nucleus (bulk). Otherwise returns
    a copy whose ``wfc.coeffs`` are multiplied by :func:`pw_taper_window` in
    ``q = |G+k|`` (the physical wavevector: at a nonzero twist the sphere is
    centred on ``-k``) with ``g_end = sqrt(2 ecutwfc)`` (QE's sphere edge,
    identical for every twist) unless the caller passes the smaller edge its
    representation keeps (``g_end=g_cut`` for a truncated ``pw`` basis), and
    ``g_start = PW_TAPER_START_FRACTION * g_end``, plus the conversion metadata
    that the Hamiltonian file records. At Gamma the window depends on ``|G|``
    only, so a ``gamma_only`` set keeps ``c(-G) = conj(c(G))``.

    Every PW-carrying projection (``pw``, ``bspline`` and the PW parts of the
    composite kinds) consumes the returned reference, so their fidelity
    diagnostics compare against the tapered bands and the taper's own cost is
    reported here as the per-band spectral loss ``sum|c|^2 (1 - w^2) / sum|c|^2``:
    logged band by band and returned as
    ``pw_taper_spectral_loss_per_band<label>`` together with the occupied mean
    and maximum ``..._mean_occupied<label>`` / ``..._max_occupied<label>``
    (``label`` names the spin channel / twist, e.g. ``'_dn'``, ``'_ik3'``, so
    every channel keeps its own summary). A ``g_end`` that retains no plane
    wave beyond the constant (``ecutwfc = 0``) has no edge and returns the
    reference unchanged.
    """
    empty = max_empty_distance(reference.cell.a, reference.atom_positions)
    if empty <= PW_TAPER_VACUUM_RADIUS_BOHR:
        logger.info(
            "PW G-sphere taper not applied: no vacuum region (largest nucleus-free distance %.2f bohr <= %.1f bohr).",
            empty,
            PW_TAPER_VACUUM_RADIUS_BOHR,
        )
        return reference, None
    wfc = reference.wfc
    g_sphere = float(np.sqrt(2.0 * reference.ecutwfc))
    g_end = g_sphere if g_end is None else min(float(g_end), g_sphere)
    if not g_end > 0.0:
        # A zero-radius sphere keeps only the constant plane wave: there is no
        # edge to smooth, so the coefficients are passed through unchanged.
        logger.info(
            "PW G-sphere taper not applied%s: the retained sphere has no edge (g_end = %.3g).",
            f" [{label.strip('_')}]" if label else "",
            g_end,
        )
        return reference, None
    g_start = float(PW_TAPER_START_FRACTION) * g_end
    q = np.linalg.norm(wfc.g_vectors(reference.cell) + wfc.k_point[None, :], axis=1)
    w = pw_taper_window(q, g_start, g_end)
    coeffs = wfc.coeffs * w[None, :]
    power = np.real(np.einsum("ng,ng->n", wfc.coeffs.conj(), wfc.coeffs))
    loss = np.real(np.einsum("ng,ng->n", wfc.coeffs.conj(), wfc.coeffs * (1.0 - w * w)[None, :])) / np.where(
        power > 0.0, power, 1.0
    )
    occ = np.asarray(reference.bands.occupations, dtype=np.float64)[: loss.size]
    occupied = occ > 0.0
    loss_occ = loss[occupied] if occupied.any() else loss
    weights = occ[occupied] if occupied.any() else np.ones_like(loss)
    loss_mean = float(np.sum(weights * loss_occ) / np.sum(weights))
    loss_max = float(np.max(loss_occ))
    logger.info(
        "PW G-sphere taper applied%s: vacuum region found (largest nucleus-free distance %.2f bohr > %.1f bohr); "
        "w(|G+k|) = 1 up to %.3f bohr^-1 and 0 at %.3f bohr^-1 (sphere edge %.3f; %d of %d plane waves "
        "attenuated); occupied-band spectral loss mean %.3e, max %.3e.",
        f" [{label.strip('_')}]" if label else "",
        empty,
        PW_TAPER_VACUUM_RADIUS_BOHR,
        g_start,
        g_end,
        g_sphere,
        int(np.count_nonzero(w < 1.0)),
        int(w.size),
        loss_mean,
        loss_max,
    )
    eigenvalues = np.asarray(reference.bands.eigenvalues, dtype=np.float64)
    for band, value in enumerate(loss):
        logger.info(
            "  PW taper band %3d%s: eigenvalue %+10.5f Ha, occupation %.3f, spectral loss %.3e",
            band,
            f" [{label.strip('_')}]" if label else "",
            eigenvalues[band] if band < eigenvalues.size else float("nan"),
            occ[band] if band < occ.size else float("nan"),
            value,
        )
    tapered = QEReference(
        cell=reference.cell,
        atom_labels=reference.atom_labels,
        atom_positions=reference.atom_positions,
        ecutwfc=reference.ecutwfc,
        fft_grid=reference.fft_grid,
        bands=QEBands(
            eigenvalues=reference.bands.eigenvalues,
            occupations=reference.bands.occupations,
            n_electrons=reference.bands.n_electrons,
            fermi_energy=reference.bands.fermi_energy,
        ),
        wfc=QEWavefunction(millers=wfc.millers, coeffs=coeffs, gamma_only=wfc.gamma_only, k_point=wfc.k_point),
        atom_elements=reference.atom_elements,
    )
    meta = {
        "pw_taper_start_fraction": float(PW_TAPER_START_FRACTION),
        "pw_taper_g_start": g_start,
        "pw_taper_g_end": g_end,
        "pw_taper_vacuum_radius_bohr": float(PW_TAPER_VACUUM_RADIUS_BOHR),
        "pw_taper_max_empty_distance_bohr": float(empty),
        f"pw_taper_spectral_loss_mean_occupied{label}": loss_mean,
        f"pw_taper_spectral_loss_max_occupied{label}": loss_max,
        f"pw_taper_spectral_loss_per_band{label}": loss.astype(np.float64),
    }
    return tapered, meta


# ---------------------------------------------------------------------------
# Fourier-space pGTO sampling (analytic FT on the QE reciprocal lattice)
# ---------------------------------------------------------------------------
# A Cartesian Gaussian primitive
#
#     g(r) = (x-R_x)^nx (y-R_y)^ny (z-R_z)^nz exp(-alpha |r-R|^2)
#
# has the closed-form (continuous) Fourier transform
#   hat g(G) = integral g(r) exp(-i G.r) dr
#            = exp(-i G.R) * (pi/alpha)^(3/2) * exp(-|G|^2 / (4 alpha))
#              * prod_axis [ i^n_axis p_{n_axis}(G_axis) ]
# where, with beta = 1/(2 alpha), the per-axis polynomial p_n is the n-th
# derivative factor of the Gaussian envelope (p_0=1, p_{m+1}=p_m' - beta G p_m).
# The periodic Gaussian's Fourier-series coefficient on a reciprocal-lattice
# vector G is simply ``tilde g(G) = hat g(G) / Omega`` (Omega = cell volume) --
# evaluating the *single-cell* transform on the reciprocal lattice automatically
# captures the periodic image sum, so there is no convergence loop, no r_cut,
# and no image_tol.


def _axis_poly_factor(n: int, g_axis: npt.NDArray[np.float64], beta: float) -> npt.NDArray[np.complex128]:
    """Per-axis factor ``i^n p_n(g_axis)`` of the Cartesian-Gaussian FT.

    ``p_n`` is the polynomial from the n-th derivative of ``exp(-beta/2 * G^2)``
    envelope bookkeeping (``beta = 1/(2 alpha)``); multiplied by ``i^n`` it is the
    contribution of one Cartesian angular exponent ``n`` to ``hat g(G)``. Closed
    forms for ``n = 0..4`` (f-shell needs at most ``n = 3``; ``n = 4`` kept for
    parity with the real-space kernel's ``_MAX_L = 4``).
    """
    g = np.asarray(g_axis, dtype=np.float64)
    if n == 0:
        return np.ones_like(g, dtype=np.complex128)
    if n == 1:
        return -1j * beta * g
    if n == 2:
        return (beta - (beta**2) * g**2).astype(np.complex128)
    if n == 3:
        return -1j * (3.0 * beta**2 * g - beta**3 * g**3)
    if n == 4:
        return (3.0 * beta**2 - 6.0 * beta**3 * g**2 + beta**4 * g**4).astype(np.complex128)
    raise ValueError(f"Cartesian angular exponent n={n} exceeds the supported maximum (4).")


def build_gto_pw_matrix(
    reference: QEReference,
    prims: list[CartPrimitive],
    k_cart: npt.NDArray[np.float64] | tuple[float, float, float] | None = None,
    millers: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.complex128]:
    r"""Analytic Fourier coefficients of every primitive on a reciprocal lattice.

    Returns ``G_full`` of shape ``(n_pw, n_prims) complex128``. Row ``g``
    corresponds to ``q_g = millers[g] @ cell.b + k_cart`` and the matrix
    holds ``sqrt(Omega) * tilde_g_mu(q_g)``, where ``Omega`` is the cell
    volume and ``tilde_g_mu`` is the primitive's cell-normalized Fourier-series
    coefficient.

    Args:
        reference: Loaded QE reference (supplies the cell). When ``millers`` is
            ``None`` the QE wfc Miller indices are used.
        prims: All-primitive Cartesian pGTO basis.
        k_cart: Bloch twist in Cartesian ``1/bohr`` units. ``None`` (or zero
            vector) selects the Gamma path. A non-Gamma QE band
            ``psi_nk(r) = sum_G c_nG e^{i(G+k).r} / sqrt(Omega)`` requires
            the design matrix to be evaluated at ``q = G + k``, so
            ``k_cart`` is added to every ``G``. The QE-stored
            ``wfc.k_point`` (xk attr) is already in ``1/bohr`` Cartesian,
            so the caller forwards it as-is (see ``qe_to_jqmc.py``).
        millers: Override the reciprocal index set. Use to evaluate pGTO Fourier
            coefficients on the full FFT cube (see :func:`fft_cube_millers`)
            instead of the (smaller) wfc G-set.
    """
    cell = reference.cell
    if millers is None:
        millers = reference.wfc.millers
    # Default ``k_cart`` to the wfc's stored
    # ``k_point`` (QE writes ``xk`` already in ``1/bohr`` Cartesian, so the
    # value is consumed verbatim). The single-twist Gamma path keeps
    # ``k_cart = (0, 0, 0)`` and the addition below is a no-op (byte-
    # identical to the plain form ``millers @ cell.b``). An explicit
    # ``k_cart=`` argument overrides this default -- used by tests that
    # exercise the projection at synthetic shifts without rebuilding
    # ``QEReference``.
    if k_cart is None:
        k_cart_arr = np.asarray(reference.wfc.k_point, dtype=np.float64).reshape(3)
    else:
        k_cart_arr = np.asarray(k_cart, dtype=np.float64).reshape(3)
    g_vectors = millers.astype(np.float64) @ cell.b  # (n_pw, 3) Cartesian
    if np.any(k_cart_arr != 0.0):
        g_vectors = g_vectors + k_cart_arr  # q = G + k
    n_pw_local = g_vectors.shape[0]
    n_prims = len(prims)
    out = np.zeros((n_pw_local, n_prims), dtype=np.complex128)
    if n_pw_local == 0 or n_prims == 0:
        return out

    gx, gy, gz = g_vectors[:, 0], g_vectors[:, 1], g_vectors[:, 2]
    g2 = gx * gx + gy * gy + gz * gz  # (n_pw_local,)
    inv_sqrt_vol = 1.0 / np.sqrt(cell.volume)

    # Group columns by exponent to reuse the (n_pw_local,) Gaussian envelope
    # exp(-|G|^2 / 4 alpha); a small dict keyed on the float exponent.
    groups: dict[float, list[int]] = {}
    for j, p in enumerate(prims):
        groups.setdefault(p.exponent, []).append(j)

    # Per-centre structure phase exp(-i G.R) reused across all primitives on
    # the same centre.
    phase_cache: dict[tuple[float, float, float], npt.NDArray[np.complex128]] = {}

    for alpha, cols in groups.items():
        beta = 1.0 / (2.0 * alpha)
        envelope = inv_sqrt_vol * (np.pi / alpha) ** 1.5 * np.exp(-g2 / (4.0 * alpha))  # (n_pw_local,)
        for j in cols:
            p = prims[j]
            nx, ny, nz = p.ang
            ax = _axis_poly_factor(nx, gx, beta)
            ay = _axis_poly_factor(ny, gy, beta)
            az = _axis_poly_factor(nz, gz, beta)
            structure_phase = phase_cache.get(p.center)
            if structure_phase is None:
                cx, cy, cz = p.center
                structure_phase = np.exp(-1j * (gx * cx + gy * cy + gz * cz))
                phase_cache[p.center] = structure_phase
            out[:, j] = envelope * ax * ay * az * structure_phase
    return out


def accumulate_gto_pw_blocks(
    reference: QEReference,
    prims: list[CartPrimitive],
    millers: npt.NDArray[np.int64],
    *,
    psi: npt.NDArray[np.complex128] | None = None,
    gram: bool = True,
    k_cart: npt.NDArray[np.float64] | tuple[float, float, float] | None = None,
) -> tuple[npt.NDArray[np.complex128] | None, npt.NDArray[np.complex128] | None]:
    r"""Accumulate pGTO Fourier Gram/RHS blocks with bounded peak memory.

    ``build_gto_pw_matrix`` is evaluated on at most one configured-memory
    chunk of reciprocal rows at a time. This bounds the production peak to
    ``O(chunk * n_gto + n_gto^2)``. ``psi`` is row-aligned with ``millers``
    and may be omitted when only the Gram matrix is required. ``k_cart``
    overrides the reference twist for an auxiliary reciprocal set; source-
    owned Miller rows normally leave it as ``None``.
    """
    millers = np.asarray(millers, dtype=np.int64)
    if psi is not None:
        psi = np.asarray(psi, dtype=np.complex128)
        if psi.ndim != 2 or psi.shape[0] != millers.shape[0]:
            raise ValueError(f"psi shape {psi.shape} must be (n_millers, n_band) with n_millers={millers.shape[0]}.")
    n_gto = len(prims)
    s_gg = np.zeros((n_gto, n_gto), dtype=np.complex128) if gram else None
    b_g = np.zeros((n_gto, psi.shape[1]), dtype=np.complex128) if psi is not None else None
    budget_rows = max(1, PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES // max(16 * int(n_gto), 1))
    chunk_rows = min(max(millers.shape[0], 1), budget_rows)
    for start in range(0, millers.shape[0], chunk_rows):
        stop = min(start + chunk_rows, millers.shape[0])
        g_chunk = build_gto_pw_matrix(reference, prims, k_cart=k_cart, millers=millers[start:stop])
        if s_gg is not None:
            s_gg += g_chunk.conj().T @ g_chunk
        if b_g is not None:
            b_g += g_chunk.conj().T @ psi[start:stop]
    if s_gg is not None:
        s_gg = 0.5 * (s_gg + s_gg.conj().T)
    return s_gg, b_g


def _psi_pw_matrix(
    reference: QEReference,
    band_idx: npt.NDArray[np.int64],
) -> npt.NDArray[np.complex128]:
    """QE band coefficients in PW space, shape ``(n_pw_QE, n_band)``.

    ``psi[g, n] = c_{G_g, n}``: the expansion coefficients of the
    cell-normalized QE band ``psi_n = Omega^{-1/2} sum_G c_{G,n} e^{i(G+k).r}``
    on the cell-normalized plane waves ``e^{i(G+k).r} / sqrt(Omega)`` -- the
    same basis in which :func:`build_gto_pw_matrix` expands the pGTO columns.
    ``G^H psi`` and ``sum_g |psi|^2`` therefore directly assemble the
    projection right-hand side and band norms (Plancherel:
    ``sum_G a*(G) b(G) = integral_Omega a* b dr``), and the projected MOs are
    in the standard unit-cell-normalized convention (a ``sqrt(Omega) * c``
    right-hand side would give every projected MO a common ``sqrt(Omega)``
    factor). No inverse FFT -- ``psi`` is already in PW space.
    """
    # wfc.coeffs is (n_band, n_g); select bands and transpose.
    psi = reference.wfc.coeffs[band_idx, :].T
    return np.asarray(psi, dtype=np.complex128)


def _cell_normalization(reference: QEReference) -> float:
    """``Omega^{-1/2}``: maps the QE amplitudes ``sum_G c_G e^{iG.r}`` to unit-cell-normalized bands.

    QE stores ``c_G`` with ``sum_G |c_G|^2 = 1`` for the cell-normalized band
    ``Omega^{-1/2} sum_G c_G e^{i(G+k).r}``. :func:`reconstruct_pw_on_grid` and
    the raw ``c_G`` used as coefficients of the unit-amplitude PW AOs both
    omit that factor; every converter output multiplies by this value so the
    emitted MOs satisfy ``integral_Omega |psi_n|^2 = 1``.
    """
    return float(1.0 / np.sqrt(float(reference.cell.volume)))


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def _project_from_S_b(
    S: npt.NDArray[np.complex128],
    b: npt.NDArray[np.complex128],
    psi_norm2: npt.NDArray[np.float64],
    svd_rcond: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], int, float, float, float]:
    """Canonical-orth projection from a precomputed overlap ``S`` and rhs ``b``.

    ``S = Phi^H Phi dV`` (Hermitian, ``(n_basis, n_basis)``) and ``b = Phi^H psi
    dV`` (``(n_basis, n_bnd)``). ``psi_norm2 = <psi_n|psi_n> dV`` is used to
    normalise the projected residual. This is the engine shared by the explicit
    ``_project_bands`` path and the FFT-based ``_mixed_result_from_design`` path
    that never materialises the PW design matrix.
    """
    s = 0.5 * (S + S.conj().T)
    w, U = np.linalg.eigh(s)
    w_max = float(w.max())
    w_min = float(w.min())
    keep = w > svd_rcond * w_max
    n_dropped = int((~keep).sum())
    w_inv = np.zeros_like(w)
    w_inv[keep] = 1.0 / w[keep]
    s_pinv = (U * w_inv) @ U.conj().T

    c = s_pinv @ b  # (n_basis, n_bnd)
    projected = np.real(np.einsum("ij,ij->j", c.conj(), s @ c))  # <psi|P|psi>
    residuals = 1.0 - projected / psi_norm2

    s_cond = float(w_max / w[keep].min()) if keep.any() else float("inf")
    s_cond_full = float(w_max / w_min) if w_min > 0 else float("inf")
    coeffs = c.T.astype(np.complex128)  # (n_bnd, n_basis)
    return residuals.astype(np.float64), coeffs, n_dropped, s_cond, s_cond_full, w_min


def _mixed_full_overlap_condition(
    S_gg: npt.NDArray[np.complex128],
    S_gp: npt.NDArray[np.complex128],
    pw_overlap: float | npt.NDArray[np.float64],
) -> float:
    """Condition number of the raw overlap, compressing its equal PW diagonal entries."""
    n_gto, n_pw = S_gp.shape
    diagonal = np.broadcast_to(pw_overlap, (n_pw,))
    if n_gto == 0 and n_pw:
        eig_min = float(diagonal.min())
        return float(diagonal.max() / eig_min) if eig_min > 0.0 else float("inf")

    omitted_eigenvalue = None
    if n_pw > n_gto:
        values, counts = np.unique(diagonal, return_counts=True)
        largest = int(np.argmax(counts))
        if counts[largest] > n_gto:
            # Rotate only the degenerate subspace: all PWs at general k, or
            # the cosine/sine columns at Gamma. The Gamma constant stays explicit.
            omitted_eigenvalue = float(values[largest])
            bulk = diagonal == omitted_eigenvalue
            coupled = S_gp if counts[largest] == n_pw else S_gp[:, bulk]
            _, r = np.linalg.qr(coupled.conj().T, mode="reduced")
            S_gp = np.concatenate((S_gp[:, ~bulk], r.conj().T), axis=1)
            diagonal = np.concatenate((diagonal[~bulk], np.full(n_gto, omitted_eigenvalue)))
    reduced = np.block(
        [
            [S_gg, S_gp],
            [S_gp.conj().T, np.diag(diagonal)],
        ]
    )
    eig = np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T))
    eig_min = float(eig[0])
    eig_max = float(eig[-1])
    if omitted_eigenvalue is not None:
        eig_min = min(eig_min, omitted_eigenvalue)
        eig_max = max(eig_max, omitted_eigenvalue)
    return float(eig_max / eig_min) if eig_min > 0.0 else float("inf")


def _gamma_real_pw_basis_map(
    millers: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Transform Gamma ``exp(iG.r)`` columns to the emitted cos/sin AOs.

    The two sparse source indices and weights for each emitted column are
    returned together with ``diagonal_scale``, where
    ``S_real_pw = V * diag(diagonal_scale)``.  The retained Gamma Miller set
    must contain both members of every nonzero ``(G, -G)`` pair.
    """
    millers = np.asarray(millers, dtype=np.int64)
    index_of = {tuple(m.tolist()): i for i, m in enumerate(millers)}
    visited = np.zeros(millers.shape[0], dtype=bool)
    source_indices: list[tuple[int, int]] = []
    source_weights: list[tuple[complex, complex]] = []
    diagonal_scale: list[float] = []
    for i, m in enumerate(millers):
        if visited[i]:
            continue
        visited[i] = True
        if np.all(m == 0):
            source_indices.append((i, i))
            source_weights.append((1.0, 0.0))
            diagonal_scale.append(1.0)
            continue
        neg = tuple((-m).tolist())
        j = index_of.get(neg, -1)
        if j < 0 or visited[j]:
            raise ValueError(f"Gamma PW condition requires a complete (G, -G) pair; missing partner for {tuple(m)}.")
        visited[j] = True
        representative, partner = (j, i) if tuple(m.tolist()) < neg else (i, j)
        source_indices.append((representative, partner))
        source_weights.append((0.5, 0.5))
        diagonal_scale.append(0.5)
        source_indices.append((representative, partner))
        source_weights.append((-0.5j, 0.5j))
        diagonal_scale.append(0.5)
    return (
        np.asarray(source_indices, dtype=np.int32).reshape((-1, 2)),
        np.asarray(source_weights, dtype=np.complex128).reshape((-1, 2)),
        np.asarray(diagonal_scale, dtype=np.float64),
    )


def _generated_pgto_overlap_condition(
    reference: QEReference,
    prims: list[CartPrimitive] | tuple[CartPrimitive, ...],
    pw_millers: npt.NDArray[np.int64] | None = None,
) -> float:
    """Raw ``cond(S)`` for the pGTO or pGTO+PW AOs emitted by the converter.

    jQMC's Cartesian pGTO leaf includes its primitive normalisation in the AO
    definition.  This function therefore applies that same factor to the
    analytic pGTO--PW cross block and uses the analytic periodic pGTO overlap.
    It does not diagonally rescale or otherwise post-normalise ``S``.
    """
    prims = tuple(prims)
    s_gg = periodic_pgto_overlap_matrix(reference, prims)
    if pw_millers is None or len(pw_millers) == 0:
        eig = np.linalg.eigvalsh(s_gg)
        eig_min = float(eig[0])
        return float(eig[-1] / eig_min) if eig_min > 0.0 else float("inf")

    pw_millers = np.asarray(pw_millers, dtype=np.int64)
    g_low = build_gto_pw_matrix(reference, list(prims), millers=pw_millers)
    normalisations = cart_primitive_normalisations(prims)
    volume = float(reference.cell.volume)
    s_gp = normalisations[:, None] * (np.sqrt(volume) * g_low.conj().T)
    if not np.any(np.asarray(reference.wfc.k_point, dtype=np.float64) != 0.0):
        source_indices, source_weights, diagonal_scale = _gamma_real_pw_basis_map(pw_millers)
        s_gp = np.sum(s_gp[:, source_indices] * source_weights[None, :, :], axis=2)
        return _mixed_full_overlap_condition(s_gg, s_gp, volume * diagonal_scale)
    return _mixed_full_overlap_condition(s_gg, s_gp, volume)


def _project_from_blocks(
    S_gg: npt.NDArray[np.complex128],
    S_gp: npt.NDArray[np.complex128],
    V: float,
    b_g: npt.NDArray[np.complex128],
    b_p: npt.NDArray[np.complex128],
    psi_norm2: npt.NDArray[np.float64],
    svd_rcond: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    int,
    float,
    float,
    float,
]:
    """Schur-complement projector for the mixed pGTO + uniform-PW basis.

    Exploits the block structure of the assembled overlap

        S = [ S_gg     S_gp ]   ,   b = [ b_g ]
            [ S_gp^H   V * I ]           [ b_p ]

    where ``S_pp = V * I`` because the plane waves are mutually orthogonal
    on the uniform FFT grid up to the cell volume. The Schur complement
    eliminates the (large) PW block analytically:

        d_pw   =  (b_p - S_gp^H * d_gto) / V
        M      =  S_gg - (1/V) * S_gp * S_gp^H          (n_gto x n_gto)
        rhs    =  b_g  - (1/V) * S_gp * b_p
        M * d_gto  =  rhs                               (solve with eigh + svd_rcond)

    For typical mixed-basis projections n_gto << n_pw, so the
    n_gto x n_gto eigh is essentially free while the
    ``_project_from_S_b`` path pays O((n_gto + n_pw)^3) on the full
    matrix -- e.g. methane at the 25 Ry auto cap has n_pw ~ 7000 and the
    full path can take tens of minutes / GBs of RAM, while the Schur
    path finishes in milliseconds.

    Returns ``(residuals, coeff_gto, coeff_pw, n_dropped, s_cond,
    s_cond_full, s_eig_min)`` -- mirrors :func:`_project_from_S_b`'s
    diagnostics, but separates the pGTO and PW coefficient blocks so the
    caller does not need to know the assembled column count.

    The PW block contributes no rank deficiency (S_pp = V * I is always
    full rank), so the reported diagnostics refer to the Schur complement
    ``M``. Public converter results replace ``s_cond_full`` with the raw full
    condition of the emitted AO basis.
    """
    n_gto = S_gg.shape[0]
    n_pw = S_gp.shape[1]
    if S_gp.shape[0] != n_gto:
        raise ValueError(f"S_gp shape {S_gp.shape} incompatible with S_gg shape {S_gg.shape}.")
    if b_g.shape[0] != n_gto:
        raise ValueError(f"b_g shape {b_g.shape} does not match n_gto={n_gto}.")
    if b_p.shape[0] != n_pw:
        raise ValueError(f"b_p shape {b_p.shape} does not match n_pw={n_pw}.")
    if b_g.shape[1] != b_p.shape[1]:
        raise ValueError(f"band-axis mismatch: b_g={b_g.shape}, b_p={b_p.shape}.")
    if V <= 0.0:
        raise ValueError(f"V (cell volume) must be positive, got {V}.")

    n_band = b_g.shape[1]

    # Schur complement M = S_gg - (1/V) * S_gp * S_gp^H. Cost
    # O(n_gto^2 * n_pw); cheap because n_gto is tiny.
    M = S_gg - (S_gp @ S_gp.conj().T) / V
    M = 0.5 * (M + M.conj().T)  # symmetrise against fp rounding

    # Modified rhs r = b_g - (1/V) * S_gp * b_p, shape (n_gto, n_band).
    rhs = b_g - (S_gp @ b_p) / V

    # Solve M * d_gto = rhs with canonical orthogonalisation / svd_rcond
    # truncation, same convention as _project_from_S_b. eigh on a 15x15
    # Hermitian matrix is microseconds.
    w, U = np.linalg.eigh(M)
    w_max = float(w.max()) if w.size else 0.0
    w_min = float(w.min()) if w.size else 0.0
    if w.size:
        keep = w > svd_rcond * w_max
    else:
        keep = np.zeros(0, dtype=bool)
    n_dropped = int((~keep).sum()) if w.size else 0
    w_inv = np.zeros_like(w)
    w_inv[keep] = 1.0 / w[keep]
    M_pinv = (U * w_inv) @ U.conj().T
    d_gto = M_pinv @ rhs  # (n_gto, n_band)

    # Reconstruct the PW coefficients analytically.
    d_pw = (b_p - S_gp.conj().T @ d_gto) / V  # (n_pw, n_band)

    # Projected norm <psi|P psi>:
    #   <P psi | P psi>  =  d^H S d
    #   = d_g^H S_gg d_g + d_g^H S_gp d_p + d_p^H S_gp^H d_g + V d_p^H d_p
    # All terms cheap given the small d_gto and the diagonal S_pp.
    Sgg_d = S_gg @ d_gto  # (n_gto, n_band)
    Sgp_dp = S_gp @ d_pw  # (n_gto, n_band)
    proj = np.real(np.einsum("ij,ij->j", d_gto.conj(), Sgg_d))
    proj += 2.0 * np.real(np.einsum("ij,ij->j", d_gto.conj(), Sgp_dp))
    proj += V * np.real(np.einsum("ij,ij->j", d_pw.conj(), d_pw))
    residuals = 1.0 - proj / psi_norm2

    s_cond = float(w_max / w[keep].min()) if keep.any() else float("inf")
    s_cond_full = float(w_max / w_min) if w_min > 0.0 else float("inf")
    return (
        residuals.astype(np.float64),
        d_gto.T.astype(np.complex128),  # (n_band, n_gto)
        d_pw.T.astype(np.complex128),  # (n_band, n_pw)
        n_dropped,
        s_cond,
        s_cond_full,
        w_min,
    )


def _project_bands(
    phi: npt.NDArray[np.complex128],
    psi: npt.NDArray[np.complex128],
    dV: float,
    svd_rcond: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], int, float, float, float]:
    """Project columns of ``psi`` onto span(``phi``); return residuals + coeffs.

    Returns ``(residuals, coeffs, n_dropped, s_cond, s_cond_full, s_eig_min)``
    where ``coeffs`` has shape ``(n_band, n_basis)``: row ``n`` is the pGTO
    expansion of band ``n`` (the conversion output).
    """
    S = (phi.conj().T @ phi) * dV
    b = (phi.conj().T @ psi) * dV
    psi_norm2 = np.real(np.einsum("ij,ij->j", psi.conj(), psi)) * dV
    return _project_from_S_b(S, b, psi_norm2, svd_rcond)


def _project_bands_pw(
    reference: QEReference,
    prims: list[CartPrimitive],
    psi: npt.NDArray[np.complex128],
    svd_rcond: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], int, float, float, float]:
    """Fit bands on the QE reciprocal lattice with bounded Fourier blocks.

    The analytic Fourier blocks and ``psi`` use cell-normalized plane waves,
    so the reciprocal-space Gram and RHS need no quadrature factor.
    """
    S, b = accumulate_gto_pw_blocks(reference, prims, reference.wfc.millers, psi=psi)
    psi_norm2 = np.real(np.einsum("ij,ij->j", psi.conj(), psi))
    return _project_from_S_b(S, b, psi_norm2.astype(np.float64), svd_rcond)


def _gto_fourier_at_millers(
    phi_gto: npt.NDArray[np.complex128],
    fft_grid: tuple[int, int, int],
    millers: npt.NDArray[np.int64],
    chunk_size: int = 8,
) -> npt.NDArray[np.complex128]:
    """FFT each column of ``phi_gto`` and sample at the given Miller indices.

    Returns ``F[h] = sum_i phi_gto[i, col] exp(-i G_h . r_i)`` for every
    requested Miller index ``h``, shape ``(n_pw, n_gto)``. Columns are
    processed in ``chunk_size`` batches so the transient FFT buffer stays
    bounded; the full dense PW design matrix is never built.
    """
    n_grid, n_gto = phi_gto.shape
    n_pw = millers.shape[0]
    n1, n2, n3 = fft_grid
    if n_grid != n1 * n2 * n3:
        raise ValueError(f"phi_gto grid size {n_grid} does not match fft_grid {(n1, n2, n3)} product {n1 * n2 * n3}.")
    m = np.array(fft_grid, dtype=int)
    idx = (millers % m).astype(int)  # (n_pw, 3), maps negative h to FFT wrap-around

    out = np.empty((n_pw, n_gto), dtype=np.complex128)
    for start in range(0, n_gto, chunk_size):
        end = min(start + chunk_size, n_gto)
        # phi_gto[:, start:end] is a column block of shape (n_grid, K). The
        # transpose-and-reshape into (K, n1, n2, n3) requires a contiguous
        # copy; ascontiguousarray makes the copy explicit (and bounded by
        # K * n_grid * 16 B) so the peak buffer stays predictable.
        cube = np.ascontiguousarray(phi_gto[:, start:end].T).reshape(end - start, n1, n2, n3)
        fft = _spfft.fftn(cube, axes=(1, 2, 3), workers=PW_PGTO_N_WORKERS)
        out[:, start:end] = fft[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    return out


def _wfc_indices_for_millers(
    wfc_millers: npt.NDArray[np.int64],
    sel_millers: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Find each row of ``sel_millers`` in ``wfc_millers`` (exact-tuple match).

    Used to pull the QE-stored PW coefficients for the retained subset of
    G-vectors, replacing a dense ``phi_pw^H psi`` matvec.
    """
    lookup = {(int(m[0]), int(m[1]), int(m[2])): i for i, m in enumerate(wfc_millers)}
    out = np.empty(sel_millers.shape[0], dtype=np.int64)
    for i, m in enumerate(sel_millers):
        key = (int(m[0]), int(m[1]), int(m[2]))
        if key not in lookup:
            raise KeyError(
                f"Selected Miller {key} not present in wfc.millers; the PW subset must be a subset of the stored G-set."
            )
        out[i] = lookup[key]
    return out


def project_pw_to_pgto(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
    alpha_min: float | None = None,
) -> ProjectionResult:
    r"""Project the QE reference's (occupied) bands onto the all-primitive pGTO basis.

    Args:
        reference: Loaded QE reference (cell, fft_grid, wfc, bands).
        shells_per_atom: Per-atom fitted shells (e.g. from
            :meth:`jqmc.upf_wrapper.ExponentInitStrategy.shells_for_element`).
        occ_only: Project only occupied bands (else all bands).
        occ_tol: Occupation threshold defining 'occupied'.
        svd_rcond: Relative eigenvalue floor for canonical orthogonalisation.
        image_tol: Unused (kept for signature parity with the ``*_debug``
            real-space path); the Fourier transform needs no image tolerance.
        alpha_min: Diffuse-pGTO exponent floor. Primitives with exponent <
            ``alpha_min`` are dropped. ``None`` keeps every primitive.

    Returns:
        ProjectionResult with per-band residuals and V1/V2/V3 diagnostics.
    """
    del image_tol  # no real-space image sum in the Fourier path
    if alpha_min is not None and alpha_min > 0.0:
        shells_per_atom = filter_shells_by_alpha_min(shells_per_atom, alpha_min)
    prims = build_pgto_primitives(shells_per_atom, reference.atom_positions)

    if occ_only:
        band_idx = reference.bands.occupied_band_indices(occ_tol)
    else:
        band_idx = np.arange(reference.wfc.n_bands, dtype=np.int64)
    occ = reference.bands.occupations[band_idx]

    logger.info(
        "PW->pGTO (Fourier): n_pw_QE=%d, alpha_min=%s, %d pGTO primitives.",
        reference.wfc.n_gvectors,
        "off" if alpha_min in (None, 0.0) else f"{alpha_min:.4f}",
        len(prims),
    )
    psi_local = _psi_pw_matrix(reference, band_idx)
    logger.info("  Projecting %d bands onto %d pGTO in PW space (G^H G + eigh)...", band_idx.size, len(prims))
    residuals, coeffs, n_dropped, s_cond, _, s_eig_min = _project_bands_pw(reference, prims, psi_local, svd_rcond)
    s_cond_full = _generated_pgto_overlap_condition(reference, prims)

    logger.info(
        "PW->pGTO projection: %d bands, %d pGTO columns, mean residual %.3e, max residual %.3e, "
        "generated-AO full cond(S) %.2e, dropped %d.",
        band_idx.size,
        len(prims),
        float(np.sum(occ * residuals) / np.sum(occ)) if occ.sum() > 0 else float(np.mean(residuals)),
        float(np.max(residuals)),
        s_cond_full,
        n_dropped,
    )
    return ProjectionResult(
        residuals=residuals,
        band_indices=band_idx,
        occupations=occ,
        coefficients=coeffs,
        primitives=tuple(prims),
        n_gto=len(prims),
        n_pw=0,
        n_dropped=n_dropped,
        s_cond=s_cond,
        s_cond_full=s_cond_full,
        s_eig_min=s_eig_min,
    )


def project_pw_to_pgto_debug(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
    alpha_min: float | None = None,
) -> ProjectionResult:
    r"""Real-space reference twin of :func:`project_pw_to_pgto` (numpy, no JAX).

    Slower and memory-heavy (``n_grid x n_gto``); retained only as the
    independent cross-check for the production path. The two are not
    bit-identical: the Fourier path measures the overlap on the band-limited QE
    reciprocal lattice while this integrates the pGTO self-overlap on the dense
    grid. See the ``test_fourier_vs_real_*`` tests.
    """
    if alpha_min is not None and alpha_min > 0.0:
        shells_per_atom = filter_shells_by_alpha_min(shells_per_atom, alpha_min)
    prims = build_pgto_primitives(shells_per_atom, reference.atom_positions)

    if occ_only:
        band_idx = reference.bands.occupied_band_indices(occ_tol)
    else:
        band_idx = np.arange(reference.wfc.n_bands, dtype=np.int64)
    occ = reference.bands.occupations[band_idx]

    cell = reference.cell
    fft_grid = reference.fft_grid
    grid = grid_points(cell, fft_grid)
    dV = cell.volume / grid.shape[0]
    logger.info(
        "PW->pGTO (real-space debug): grid n_points=%d, alpha_min=%s, %d pGTO primitives.",
        grid.shape[0],
        "off" if alpha_min in (None, 0.0) else f"{alpha_min:.4f}",
        len(prims),
    )
    logger.info("  Building pGTO design matrix...")
    # u-gauge at non-Gamma: ``reconstruct_pw_orbitals`` returns the
    # periodic part (no e^{ik.r} envelope), so the design matrix must be
    # in the same gauge. Gamma path is byte-identical.
    phi = _u_gauge_design_matrix(cell, prims, grid, reference.wfc.k_point, tol=image_tol)
    logger.info("  Reconstructing PW bands on grid...")
    # Unit-cell-normalized bands, matching the Fourier path.
    psi = reconstruct_pw_orbitals(reference.wfc, fft_grid)[:, band_idx] * _cell_normalization(reference)
    logger.info("  Projecting %d bands onto %d pGTO (Gram + eigh)...", band_idx.size, len(prims))
    residuals, coeffs, n_dropped, s_cond, s_cond_full, s_eig_min = _project_bands(phi, psi, dV, svd_rcond)
    s_cond_full = _generated_pgto_overlap_condition(reference, prims)

    logger.info(
        "PW->pGTO projection (debug): %d bands, %d pGTO columns, mean residual %.3e, max residual %.3e, "
        "generated-AO full cond(S) %.2e, dropped %d.",
        band_idx.size,
        len(prims),
        float(np.sum(occ * residuals) / np.sum(occ)) if occ.sum() > 0 else float(np.mean(residuals)),
        float(np.max(residuals)),
        s_cond_full,
        n_dropped,
    )
    return ProjectionResult(
        residuals=residuals,
        band_indices=band_idx,
        occupations=occ,
        coefficients=coeffs,
        primitives=tuple(prims),
        n_gto=len(prims),
        n_pw=0,
        n_dropped=n_dropped,
        s_cond=s_cond,
        s_cond_full=s_cond_full,
        s_eig_min=s_eig_min,
    )


# ---------------------------------------------------------------------------
# Mixed pGTO + low-G plane-wave basis
# ---------------------------------------------------------------------------
def pw_gvectors_within(reference: QEReference, g_cut: float) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Retained low-energy plane waves, selected by physical ``|G+k| <= g_cut``.

    Returns ``(millers, g_vectors)`` sorted by increasing ``|G+k|`` (so a
    larger ``g_cut`` is a strict superset). Selecting from the wfc's own G set
    keeps every PW column an exact Fourier mode of the bands. Using the
    physical Bloch wavevector makes equivalent ``+pi``/``-pi`` boundary
    representations invariant when their Miller rows are transported together.
    """
    g_all = reference.wfc.g_vectors(reference.cell)  # (n_g, 3) Cartesian
    millers = reference.wfc.millers
    qnorm = np.linalg.norm(g_all + reference.wfc.k_point[None, :], axis=1)
    mask = qnorm <= g_cut
    order = np.argsort(qnorm[mask], kind="stable")
    return millers[mask][order].astype(np.int64), g_all[mask][order]


def build_pw_design_matrix(grid: npt.NDArray[np.float64], g_vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    """Plane-wave columns ``exp(i G . r)`` on ``grid``, shape ``(n_points, n_pw)``."""
    if g_vectors.shape[0] == 0:
        return np.empty((grid.shape[0], 0), dtype=np.complex128)
    phase = grid @ g_vectors.T
    return np.exp(1j * phase).astype(np.complex128)


def _select_band_indices(reference: QEReference, occ_only: bool, occ_tol: float, n_total: int) -> npt.NDArray[np.int64]:
    if occ_only:
        return reference.bands.occupied_band_indices(occ_tol)
    return np.arange(n_total, dtype=np.int64)


def _mixed_result_from_design(
    phi_gto: npt.NDArray[np.complex128],
    prims: list[CartPrimitive],
    pw_millers: npt.NDArray[np.int64],
    psi: npt.NDArray[np.complex128],
    band_idx: npt.NDArray[np.int64],
    occ: npt.NDArray[np.float64],
    dV: float,
    svd_rcond: float,
    alpha_min: float,
    g_cut: float,
    *,
    fft_grid: tuple[int, int, int],
    cell_volume: float,
    wfc_millers: npt.NDArray[np.int64],
    wfc_coeffs: npt.NDArray[np.complex128],
) -> ProjectionResult:
    """Project ``psi`` onto ``[phi_gto | phi_pw]`` and package a ProjectionResult.

    The PW block ``phi_pw[i, m] = exp(i G_m . r_i)`` is left implicit: the
    overlap and right-hand side blocks involving it are built via 3D FFT of
    ``phi_gto`` (for the cross block ``Phi_gto^H Phi_pw``) and by extracting
    the QE-stored PW coefficients at the retained Millers (for ``Phi_pw^H
    psi``). On a uniform FFT grid the PW block is diagonal,
    ``Phi_pw^H Phi_pw = V I``, so the inner overlap is built without ever
    forming the dense ``(n_grid, n_pw)`` design matrix -- this is what makes
    the converter usable when ``n_pw * n_grid * 16`` would dwarf RAM.
    """
    n_gto = phi_gto.shape[1]
    n_pw = pw_millers.shape[0]

    if n_pw == 0:
        residuals, coeffs, n_dropped, s_cond, s_cond_full, s_eig_min = _project_bands(phi_gto, psi, dV, svd_rcond)
        coeff_gto = coeffs
        coeff_pw = np.empty((coeffs.shape[0], 0), dtype=np.complex128)
    else:
        # GTO blocks (small overlap, plus the rhs against psi).
        S_gg = (phi_gto.conj().T @ phi_gto) * dV
        b_g = (phi_gto.conj().T @ psi) * dV

        # Cross block: S_gp[j, m] = <phi_gto_j | phi_pw_m> = conj(F_j[G_m]) * dV
        # where F_j is the discrete Fourier transform of column j of phi_gto.
        gto_fft = _gto_fourier_at_millers(phi_gto, fft_grid, pw_millers)
        S_gp = (gto_fft.conj() * dV).T  # (n_gto, n_pw)

        # PW rhs: <phi_pw_m | psi_n> = V * c_{G_m, n} because psi is its own
        # (``wfc_coeffs`` carries the same scale as ``psi``: the caller passes
        # ``Omega^{-1/2} c`` for the unit-cell-normalized bands)
        # Fourier series (the QE-stored coefficients) and the grid sum picks
        # out one term per G. PW-PW block is V * I (uniform-grid orthogonality)
        # and is consumed analytically by _project_from_blocks below; we do
        # not assemble it explicitly.
        wfc_pw_indices = _wfc_indices_for_millers(wfc_millers, pw_millers)
        b_p = float(cell_volume) * wfc_coeffs[np.ix_(band_idx, wfc_pw_indices)].T  # (n_pw, n_band)

        psi_norm2 = np.real(np.einsum("ij,ij->j", psi.conj(), psi)) * dV

        # Schur-complement projection: eliminates the (n_pw x n_pw) PW block
        # analytically (S_pp = V*I), so the only eigh is on the tiny
        # n_gto x n_gto Schur matrix. Beats an assemble-then-eigh
        # path by O((n_gto + n_pw)^3 / n_gto^3) ~ 10^6x at trial 8 on
        # heavy fixtures (methane n_pw ~ 7000).
        (
            residuals,
            coeff_gto,
            coeff_pw,
            n_dropped,
            s_cond,
            s_cond_full,
            s_eig_min,
        ) = _project_from_blocks(
            S_gg,
            S_gp,
            float(cell_volume),
            b_g,
            b_p,
            psi_norm2,
            svd_rcond,
        )
    return ProjectionResult(
        residuals=residuals,
        band_indices=band_idx,
        occupations=occ,
        coefficients=coeff_gto,
        primitives=tuple(prims),
        n_gto=n_gto,
        n_pw=n_pw,
        n_dropped=n_dropped,
        s_cond=s_cond,
        s_cond_full=s_cond_full,
        s_eig_min=s_eig_min,
        pw_millers=pw_millers,
        pw_coefficients=coeff_pw,
        alpha_min=alpha_min,
        g_cut=g_cut,
    )


def _mixed_result_from_pw_blocks(
    s_gg: npt.NDArray[np.complex128],
    b_g: npt.NDArray[np.complex128],
    g_gto_pw: npt.NDArray[np.complex128],
    psi_pw: npt.NDArray[np.complex128],
    psi_norm2: npt.NDArray[np.float64],
    prims: list[CartPrimitive],
    pw_millers: npt.NDArray[np.int64],
    band_idx: npt.NDArray[np.int64],
    occ: npt.NDArray[np.float64],
    svd_rcond: float,
    alpha_min: float,
    g_cut: float,
    *,
    cell_volume: float,
) -> ProjectionResult:
    """Package a mixed projection from already accumulated Fourier blocks."""
    n_gto = len(prims)
    n_pw = pw_millers.shape[0]
    if s_gg.shape != (n_gto, n_gto):
        raise ValueError(f"s_gg shape {s_gg.shape} does not match n_gto={n_gto}.")
    if b_g.shape[0] != n_gto:
        raise ValueError(f"b_g shape {b_g.shape} does not match n_gto={n_gto}.")

    if n_pw == 0:
        residuals, coeff_gto, n_dropped, s_cond, s_cond_full, s_eig_min = _project_from_S_b(
            s_gg,
            b_g,
            psi_norm2,
            svd_rcond,
        )
        coeff_pw = np.empty((coeff_gto.shape[0], 0), dtype=np.complex128)
    else:
        sqrt_v = np.sqrt(float(cell_volume))
        s_gp = sqrt_v * g_gto_pw.conj().T
        b_p = sqrt_v * psi_pw
        (
            residuals,
            coeff_gto,
            coeff_pw,
            n_dropped,
            s_cond,
            s_cond_full,
            s_eig_min,
        ) = _project_from_blocks(
            s_gg,
            s_gp,
            float(cell_volume),
            b_g,
            b_p,
            psi_norm2,
            svd_rcond,
        )
    return ProjectionResult(
        residuals=residuals,
        band_indices=band_idx,
        occupations=occ,
        coefficients=coeff_gto,
        primitives=tuple(prims),
        n_gto=n_gto,
        n_pw=n_pw,
        n_dropped=n_dropped,
        s_cond=s_cond,
        s_cond_full=s_cond_full,
        s_eig_min=s_eig_min,
        pw_millers=pw_millers,
        pw_coefficients=coeff_pw,
        alpha_min=alpha_min,
        g_cut=g_cut,
    )


def project_pw_to_mixed(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    g_cut: float,
    drop_diffuse: bool = True,
    tail_tol: float = PW_PGTO_TAIL_TOL,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
    alpha_min: float | None = None,
) -> ProjectionResult:
    r"""Project bands onto a mixed pGTO + low-G plane-wave basis at a fixed ``g_cut``.

    The projection is approximate; a small nonzero residual can remain. Check fit
    quality and the convergence of the quantities of interest.

    Args:
        reference: Loaded QE reference.
        shells_per_atom: Per-atom fitted shells.
        g_cut: Plane-wave handoff cutoff ``|G+k|`` (Bohr^-1). PWs with
            ``|G+k| <= g_cut`` are appended.
        drop_diffuse: Apply the ``alpha_min`` floor coupled to ``g_cut``.
        tail_tol: Momentum-tail tolerance for the ``alpha_min`` coupling.
        occ_only: Project only occupied bands (else all bands).
        occ_tol: Occupation threshold defining 'occupied'.
        svd_rcond: Relative eigenvalue floor for canonical orthogonalisation.
        image_tol: Unused (kept for signature parity with the ``*_debug``
            real-space path); the Fourier transform needs no image tolerance.
        alpha_min: Explicit diffuse-pGTO exponent floor. When ``None`` (default)
            the floor is the ``g_cut``-coupled ``alpha_min_from_gcut`` value (or
            ``0`` if ``drop_diffuse`` is ``False``); when given, this value
            overrides the auto-derived floor.
    """
    del image_tol  # no real-space image sum in the Fourier path
    cell = reference.cell

    if alpha_min is None:
        alpha_min = alpha_min_from_gcut(g_cut, tail_tol) if drop_diffuse else 0.0
    shells_f = filter_shells_by_alpha_min(shells_per_atom, alpha_min)
    prims = build_pgto_primitives(shells_f, reference.atom_positions)

    band_idx = _select_band_indices(reference, occ_only, occ_tol, reference.wfc.n_bands)
    occ = reference.bands.occupations[band_idx]
    pw_millers, g_sel = pw_gvectors_within(reference, g_cut)

    logger.info(
        "PW->pGTO+PW (Fourier): n_pw_QE=%d, g_cut=%.3f, drop_diffuse=%s, alpha_min=%.4f -> %d pGTO primitives, n_PW=%d.",
        reference.wfc.n_gvectors,
        g_cut,
        drop_diffuse,
        alpha_min,
        len(prims),
        g_sel.shape[0],
    )
    # Plancherel-complete pGTO overlap on the full FFT cube.  Accumulate it in
    # bounded row chunks: a dense matrix of 160^3 * 1024 complex entries is
    # ~67 GiB, before its G^H G temporary allocations.
    fft_millers = fft_cube_millers(reference.fft_grid)
    psi_local = _psi_pw_matrix(reference, band_idx)
    # The FFT cube is an auxiliary finite quadrature, not a source-owned
    # Miller set that can be transported with k.  Evaluate it in one canonical
    # reciprocal gauge so equivalent k+G source labels produce the same pGTO
    # self-overlap.  Source-owned RHS/cross blocks below keep their original
    # millers and raw k, whose physical q vectors are already invariant.
    s_gg, _ = accumulate_gto_pw_blocks(
        reference,
        prims,
        fft_millers,
        gram=True,
        k_cart=_canonical_twist_cart(reference),
    )
    _, b_g = accumulate_gto_pw_blocks(
        reference,
        prims,
        reference.wfc.millers,
        psi=psi_local,
        gram=False,
    )
    pw_global_idx = _wfc_indices_for_millers(reference.wfc.millers, pw_millers)
    g_gto_pw = build_gto_pw_matrix(reference, prims, millers=pw_millers)
    psi_pw = psi_local[pw_global_idx, :]
    psi_norm2 = np.real(np.einsum("ij,ij->j", psi_local.conj(), psi_local)).astype(np.float64)
    logger.info(
        "  Projecting %d bands onto %d (pGTO) + %d (PW) basis in PW space (S_gg over %d FFT-cube G's)...",
        band_idx.size,
        len(prims),
        g_sel.shape[0],
        fft_millers.shape[0],
    )
    res = _mixed_result_from_pw_blocks(
        s_gg,
        b_g,
        g_gto_pw,
        psi_pw,
        psi_norm2,
        prims,
        pw_millers,
        band_idx,
        occ,
        svd_rcond,
        alpha_min,
        g_cut,
        cell_volume=cell.volume,
    )
    res = replace(res, s_cond_full=_generated_pgto_overlap_condition(reference, prims, pw_millers))
    logger.info(
        "PW->pGTO+PW: g_cut=%.3f, alpha_min=%.3f, %d pGTO + %d PW columns, "
        "mean residual %.3e, max %.3e, full cond(S) %.2e, dropped %d.",
        g_cut,
        alpha_min,
        res.n_gto,
        res.n_pw,
        res.mean_residual,
        res.max_residual,
        res.s_cond_full,
        res.n_dropped,
    )
    return res


def project_pw_to_mixed_debug(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    g_cut: float,
    drop_diffuse: bool = True,
    tail_tol: float = PW_PGTO_TAIL_TOL,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
    alpha_min: float | None = None,
) -> ProjectionResult:
    r"""Real-space reference twin of :func:`project_pw_to_mixed` (numpy, no JAX).

    Retained only as the cross-check for the production path -- see the
    ``test_fourier_vs_real_*`` tests.
    """
    cell = reference.cell

    if alpha_min is None:
        alpha_min = alpha_min_from_gcut(g_cut, tail_tol) if drop_diffuse else 0.0
    shells_f = filter_shells_by_alpha_min(shells_per_atom, alpha_min)
    prims = build_pgto_primitives(shells_f, reference.atom_positions)

    band_idx = _select_band_indices(reference, occ_only, occ_tol, reference.wfc.n_bands)
    occ = reference.bands.occupations[band_idx]
    pw_millers, g_sel = pw_gvectors_within(reference, g_cut)

    grid = grid_points(cell, reference.fft_grid)
    dV = cell.volume / grid.shape[0]
    logger.info(
        "PW->pGTO+PW (real-space debug): grid n_points=%d, g_cut=%.3f, drop_diffuse=%s.", grid.shape[0], g_cut, drop_diffuse
    )
    logger.info("  alpha_min=%.4f -> %d pGTO primitives kept; building pGTO design matrix...", alpha_min, len(prims))
    # u-gauge at non-Gamma (see _u_gauge_design_matrix): matches the
    # u-gauge PW bands from ``reconstruct_pw_orbitals`` and the implicit
    # FFT PW block (e^{iG.r} columns). Gamma path is byte-identical.
    phi_gto = _u_gauge_design_matrix(cell, prims, grid, reference.wfc.k_point, tol=image_tol)
    logger.info("  PW block kept implicit (FFT-based): n_PW=%d.", g_sel.shape[0])
    logger.info("  Reconstructing PW bands on grid...")
    # Unit-cell-normalized bands; the PW right-hand side below uses
    # the same factor so both blocks stay consistent with the Fourier path.
    cell_norm = _cell_normalization(reference)
    psi = reconstruct_pw_orbitals(reference.wfc, reference.fft_grid)[:, band_idx] * cell_norm
    logger.info(
        "  Projecting %d bands onto %d (pGTO) + %d (PW) basis (Gram + eigh)...", band_idx.size, len(prims), g_sel.shape[0]
    )
    res = _mixed_result_from_design(
        phi_gto,
        prims,
        pw_millers,
        psi,
        band_idx,
        occ,
        dV,
        svd_rcond,
        alpha_min,
        g_cut,
        fft_grid=reference.fft_grid,
        cell_volume=cell.volume,
        wfc_millers=reference.wfc.millers,
        wfc_coeffs=reference.wfc.coeffs * cell_norm,
    )
    res = replace(res, s_cond_full=_generated_pgto_overlap_condition(reference, prims, pw_millers))
    logger.info(
        "PW->pGTO+PW (debug): g_cut=%.3f, alpha_min=%.3f, %d pGTO + %d PW columns, "
        "mean residual %.3e, max %.3e, full cond(S) %.2e, dropped %d.",
        g_cut,
        alpha_min,
        res.n_gto,
        res.n_pw,
        res.mean_residual,
        res.max_residual,
        res.s_cond_full,
        res.n_dropped,
    )
    return res


@dataclass(frozen=True)
class MixedBasisSweep:
    """Record of one mixed-basis sweep used to choose ``g_cut``.

    ``best`` is the selected (fewest-PW) result meeting the residual target;
    ``trials`` holds every probed result ordered by increasing ``g_cut``.
    """

    best: ProjectionResult
    trials: tuple[ProjectionResult, ...]
    target_met: bool


def optimize_mixed_basis(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    residual_target: float = 1.0e-4,
    g_cut_values: npt.NDArray[np.float64] | list[float] | None = None,
    n_gcut: int = 8,
    drop_diffuse: bool = True,
    tail_tol: float = PW_PGTO_TAIL_TOL,
    fixed_alpha_min: float | None = None,
    max_g_cut: float | None = None,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> MixedBasisSweep:
    r"""Find the smallest low-G plane-wave block reaching ``residual_target``.

    Returns the first trial meeting ``residual_target`` (fewest retained plane
    waves); if none reaches it, returns the lowest-residual trial with
    ``target_met=False`` and logs a WARNING naming the achieved residual and the
    cap.

    Args:
        reference: Loaded QE reference.
        shells_per_atom: Per-atom fitted shells.
        residual_target: Occupation-weighted mean-residual goal.
        g_cut_values: Explicit ``g_cut`` ladder. If ``None``, a geometric ladder
            of ``n_gcut`` points up to the PW kinetic cutoff
            ``g_max = sqrt(2 * ecutwfc)`` is used.
        n_gcut: Number of ladder points when ``g_cut_values`` is ``None``.
        drop_diffuse: Apply, per trial, the ``alpha_min`` floor coupled to that
            trial's ``g_cut``.
        tail_tol: Momentum-tail tolerance for the ``alpha_min`` coupling.
        fixed_alpha_min: If set, hold the pGTO exponent floor at this value
            for every trial and ignore ``drop_diffuse`` / ``tail_tol``. Default
            ``None`` keeps the ``drop_diffuse``-driven behaviour.
        max_g_cut: Hard upper bound on the ladder (Bohr^-1). When set, the
            default geometric ladder uses ``min(sqrt(2 ecutwfc), max_g_cut)``
            as its upper bound; any explicit ``g_cut_values`` larger than
            ``max_g_cut`` are dropped. ``None`` disables the cap.
        occ_only: Project only occupied bands (else all bands).
        occ_tol: Occupation threshold defining 'occupied'.
        svd_rcond: Relative eigenvalue floor for canonical orthogonalisation.
        image_tol: Unused (kept for signature parity with the ``*_debug``
            real-space path); the Fourier transform needs no image tolerance.
    """
    cell = reference.cell
    # Fourier-space sweep: overlaps are PW-space sums on the reciprocal
    # lattice; no dense FFT real-space grid is ever materialised. The
    # real-space reference path lives in the ``*_debug`` single-shot
    # projectors (no sweep variant is provided).
    del image_tol  # no real-space image sum in the Fourier path

    # ---- Build the g_cut ladder FIRST so we can derive the pre-filter --
    # floor from its smallest point when drop_diffuse=True.
    # Lower bound = 0.5 * g_max: a typical PSWFC-fitted pGTO has
    # alpha_max ~ a few -> Fourier reach g ~ sqrt(2 alpha_max) ~ 3 Bohr^-1
    # ~ 9 Ry, so PW trials below ~ that reach add only G content the pGTO
    # already covers and waste a trial. 0.25 is too generous when callers
    # also cap ``max_g_cut`` (e.g. via PW_PGTO_AUTO_MAX_ECUTWFC_RY=30):
    # the relative floor collapses to ~ 2 Ry and the first 3 trials are
    # structurally below the pGTO reach.
    if g_cut_values is None:
        g_max = float(np.sqrt(2.0 * reference.ecutwfc))
        if max_g_cut is not None:
            g_max = min(g_max, float(max_g_cut))
        g_cut_values = np.geomspace(0.5 * g_max, g_max, n_gcut)
    g_list = sorted(float(g) for g in g_cut_values)
    if max_g_cut is not None:
        # Clip explicit ladder entries past the cap so callers that pass
        # both g_cut_values and max_g_cut still respect the cap.
        clipped = [g for g in g_list if g <= float(max_g_cut)]
        if clipped:
            g_list = clipped

    # Full primitive set, possibly pre-filtered before the expensive
    # image-summed build_gto_design_matrix call. Pre-filter precedence:
    # explicit fixed_alpha_min wins (sweep-fixed floor doubles as a
    # build-cost guard); else when drop_diffuse=True, derive the loosest
    # per-trial mask from the smallest ladder point -- no trial would ever
    # have wanted the primitives below that floor, so dropping them up
    # front is free fidelity-wise and saves build time on bases with
    # super-diffuse primitives (e.g. methane aug-cc-* at alpha ~ 1e-3).
    prims_full = build_pgto_primitives(shells_per_atom, reference.atom_positions)
    pre_floor: float | None = None
    if fixed_alpha_min is not None and float(fixed_alpha_min) > 0.0:
        pre_floor = float(fixed_alpha_min)
    elif drop_diffuse and g_list:
        derived = alpha_min_from_gcut(g_list[0], tail_tol)
        if derived > 0.0:
            pre_floor = derived
    if pre_floor is not None:
        n_before = len(prims_full)
        prims_full = [p for p in prims_full if p.exponent >= pre_floor]
        n_after = len(prims_full)
        if n_after < n_before:
            logger.info(
                "optimize_mixed_basis: pre-filtered %d -> %d primitives (alpha_min=%.4f) before build_gto_design_matrix.",
                n_before,
                n_after,
                pre_floor,
            )
    # FFT-grid resolvability check: a Gaussian with exponent alpha has
    # width sigma = 1/sqrt(2 alpha). Once sigma drops below the minimum
    # grid spacing dx = min(L_d / N_d), the primitive is no longer
    # FFT-resolvable on this reference grid; its projected coefficient
    # picks up grid noise. WARN (don't drop) so the user can either
    # (a) re-run QE with a denser ecutrho, (b) lower G_target in
    # UPFExponentInit, or (c) accept the noise.
    if prims_full:
        dx_min = float(min(np.linalg.norm(cell.a[d]) / reference.fft_grid[d] for d in range(3)))
        alpha_max_safe = 0.5 / (dx_min * dx_min)
        unresolved = [p.exponent for p in prims_full if p.exponent > alpha_max_safe]
        if unresolved:
            logger.warning(
                "optimize_mixed_basis: %d primitive(s) with alpha > %.2f "
                "exceed the FFT grid's resolvable max (cell dx_min=%.4f Bohr; "
                "alpha_max_safe = 1/(2 dx_min^2) = %.2f). These primitives "
                "will project with grid-noise-level accuracy. Largest "
                "offending alpha=%.2f. Mitigations: raise QE ecutrho "
                "(denser fft_grid), lower UPFExponentInit's g_target, or accept "
                "the noise on the offending primitives.",
                len(unresolved),
                alpha_max_safe,
                dx_min,
                alpha_max_safe,
                max(unresolved),
            )

    exps_full = np.array([p.exponent for p in prims_full], dtype=np.float64)

    band_idx = _select_band_indices(reference, occ_only, occ_tol, reference.wfc.n_bands)
    psi_local = _psi_pw_matrix(reference, band_idx)
    occ = reference.bands.occupations[band_idx]
    # The auto sweep uses the wfc-bounded overlap convention.  Accumulate its
    # full primitive Gram/RHS once, then slice the small blocks per alpha mask;
    # never retain an n_G * n_pgto Fourier matrix.
    s_gg_full, b_g_full = accumulate_gto_pw_blocks(
        reference,
        prims_full,
        reference.wfc.millers,
        psi=psi_local,
        gram=True,
    )
    psi_norm2 = np.real(np.einsum("ij,ij->j", psi_local.conj(), psi_local)).astype(np.float64)

    import time as _time

    trials: list[ProjectionResult] = []
    best: ProjectionResult | None = None
    logger.info(
        "optimize_mixed_basis: sweeping %d g_cut points in [%.3f, %.3f] Bohr^-1 "
        "(= ecutwfc in [%.2f, %.2f] Ry); target=%.2e, fixed_alpha_min=%s.",
        len(g_list),
        g_list[0],
        g_list[-1],
        g_list[0] ** 2,
        g_list[-1] ** 2,
        residual_target,
        "None" if fixed_alpha_min is None else f"{fixed_alpha_min:.4g}",
    )
    for trial_idx, g_cut in enumerate(g_list):
        t_trial = _time.monotonic()
        logger.info(
            "  [trial %d/%d] starting: g_cut=%.3f Bohr^-1 (ecutwfc=%.2f Ry)...",
            trial_idx + 1,
            len(g_list),
            g_cut,
            g_cut**2,
        )
        if fixed_alpha_min is not None:
            alpha_min = float(fixed_alpha_min)
        elif drop_diffuse:
            alpha_min = alpha_min_from_gcut(g_cut, tail_tol)
        else:
            alpha_min = 0.0
        col_mask = exps_full >= alpha_min
        prims = [p for p, keep in zip(prims_full, col_mask, strict=True) if keep]

        pw_millers, g_sel = pw_gvectors_within(reference, g_cut)

        pw_global_idx = _wfc_indices_for_millers(reference.wfc.millers, pw_millers)
        col_idx = np.flatnonzero(col_mask)
        s_gg = s_gg_full[np.ix_(col_idx, col_idx)]
        b_g = b_g_full[col_idx, :]
        g_gto_pw = build_gto_pw_matrix(reference, prims, millers=pw_millers)
        res = _mixed_result_from_pw_blocks(
            s_gg,
            b_g,
            g_gto_pw,
            psi_local[pw_global_idx, :],
            psi_norm2,
            prims,
            pw_millers,
            band_idx,
            occ,
            svd_rcond,
            alpha_min,
            g_cut,
            cell_volume=cell.volume,
        )
        res = replace(res, s_cond_full=_generated_pgto_overlap_condition(reference, prims, pw_millers))
        trials.append(res)
        dt_trial = _time.monotonic() - t_trial
        logger.info(
            "  [trial %d/%d] done in %.1fs: %d pGTO + %d PW, mean residual %.3e, full cond(S) %.2e.",
            trial_idx + 1,
            len(g_list),
            dt_trial,
            res.n_gto,
            res.n_pw,
            res.mean_residual,
            res.s_cond_full,
        )
        if res.mean_residual <= residual_target:
            best = res
            break

    target_met = best is not None
    if best is None:
        # Library stays quiet here; the caller (e.g. jqmc_tool's auto branch)
        # decides whether a missed target is an error condition or just a
        # diagnostic and is responsible for the user-facing message.
        best = min(trials, key=lambda r: r.mean_residual)
    return MixedBasisSweep(best=best, trials=tuple(trials), target_met=target_met)


# ---------------------------------------------------------------------------
# Plane-wave-only conversion (PW -> PW)
# ---------------------------------------------------------------------------
def project_pw_to_pw(
    reference: QEReference,
    g_cut: float | None = None,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
) -> ProjectionResult:
    r"""Represent the bands in a truncated physical PW basis ``\|G+k\| <= g_cut``.

    The stored PW coefficients are already an orthogonal expansion, so this
    conversion is exact and grid-free: the retained coefficients are simply the
    band's own Fourier amplitudes and the residual is the fraction of spectral
    power discarded,

        r_n = sum_{\|G+k\| > g_cut} \|c_{n,G}\|^2 / sum_G \|c_{n,G}\|^2 .

    ``g_cut=None`` keeps every plane wave (residual 0, the loss-free reference).

    ``pw_coefficients`` are ``Omega^{-1/2} c_{n,G}``: the
    coefficients of the unit-amplitude PW AOs ``e^{i(G+k).r}`` for the
    unit-cell-normalized band, so ``integral_Omega |psi_n|^2 = 1 - r_n``.
    """
    wfc = reference.wfc
    qnorm = np.linalg.norm(wfc.g_vectors(reference.cell) + wfc.k_point[None, :], axis=1)
    if g_cut is None:
        mask = np.ones(qnorm.shape, dtype=bool)
        g_cut_val = float(qnorm.max())
    else:
        mask = qnorm <= g_cut
        g_cut_val = float(g_cut)
    order = np.argsort(qnorm[mask], kind="stable")
    keep_idx = np.nonzero(mask)[0][order]

    band_idx = _select_band_indices(reference, occ_only, occ_tol, wfc.n_bands)
    occ = reference.bands.occupations[band_idx]
    c = wfc.coeffs[band_idx]  # (n_band, n_g)
    power_all = np.real(np.einsum("ng,ng->n", c.conj(), c))
    c_keep = c[:, keep_idx]
    power_keep = np.real(np.einsum("ng,ng->n", c_keep.conj(), c_keep))
    residuals = 1.0 - power_keep / power_all

    logger.info(
        "PW->PW: g_cut=%.3f, %d/%d plane waves, mean residual %.3e, max %.3e.",
        g_cut_val,
        keep_idx.size,
        wfc.n_gvectors,
        float(np.sum(occ * residuals) / np.sum(occ)) if occ.sum() > 0 else float(np.mean(residuals)),
        float(np.max(residuals)) if residuals.size else float("nan"),
    )
    return ProjectionResult(
        residuals=residuals.astype(np.float64),
        band_indices=band_idx,
        occupations=occ,
        coefficients=np.empty((band_idx.size, 0), dtype=np.complex128),
        primitives=(),
        n_gto=0,
        n_pw=int(keep_idx.size),
        n_dropped=0,
        s_cond=1.0,  # orthogonal PW basis: S proportional to identity
        s_cond_full=1.0,
        s_eig_min=1.0,
        pw_millers=wfc.millers[keep_idx].astype(np.int64),
        pw_coefficients=(c_keep * _cell_normalization(reference)).astype(np.complex128),
        g_cut=g_cut_val,
    )


# ---------------------------------------------------------------------------
# Periodic uniform B-spline conversion (PW -> B-spline)
# ---------------------------------------------------------------------------
# Integer samples of the centred cardinal B-spline B_p (normalised to sum 1):
#   cubic  B_3 at offsets [-1,0,1]      = [1, 4, 1] / 6
#   quintic B_5 at offsets [-2,-1,0,1,2] = [1, 26, 66, 26, 1] / 120
_BSPLINE_INT_STENCIL: dict[int, tuple[float, ...]] = {
    3: (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0),
    5: (1.0 / 120.0, 26.0 / 120.0, 66.0 / 120.0, 26.0 / 120.0, 1.0 / 120.0),
}


def _bspline_weights(frac: npt.NDArray[np.float64], order: int) -> npt.NDArray[np.float64]:
    """Centred uniform B-spline basis weights at fractional offsets ``frac``.

    Returns ``(n, order+1)``: row ``k`` holds the weights for the ``order+1``
    control points at integer offsets ``[-(order-1)/2 .. (order+3)/2]`` relative
    to ``floor(t)`` (i.e. ``i-1..i+2`` for cubic, ``i-2..i+3`` for quintic). The
    weights are the cardinal B-spline ``B_order`` sampled at the control-point
    distances, so at ``frac=0`` they reduce to :data:`_BSPLINE_INT_STENCIL`.
    """
    f = np.asarray(frac, dtype=np.float64)
    if order == 3:
        w0 = (1.0 - f) ** 3
        w1 = 3.0 * f**3 - 6.0 * f**2 + 4.0
        w2 = -3.0 * f**3 + 3.0 * f**2 + 3.0 * f + 1.0
        w3 = f**3
        return np.stack([w0, w1, w2, w3], axis=-1) / 6.0
    if order == 5:
        w0 = (1.0 - f) ** 5
        w1 = 26.0 - 50.0 * f + 20.0 * f**2 + 20.0 * f**3 - 20.0 * f**4 + 5.0 * f**5
        w2 = 66.0 - 60.0 * f**2 + 30.0 * f**4 - 10.0 * f**5
        w3 = 26.0 + 50.0 * f + 20.0 * f**2 - 20.0 * f**3 - 20.0 * f**4 + 10.0 * f**5
        w4 = 1.0 + 5.0 * f + 10.0 * f**2 + 10.0 * f**3 + 5.0 * f**4 - 5.0 * f**5
        w5 = f**5
        return np.stack([w0, w1, w2, w3, w4, w5], axis=-1) / 120.0
    raise ValueError(f"Unsupported B-spline order {order!r} (use 3 or 5).")


def bspline_prefilter_periodic(values: npt.NDArray[np.complex128], order: int) -> npt.NDArray[np.complex128]:
    r"""Periodic B-spline interpolation coefficients for a uniform 3D grid.

    Returns coefficients with the same grid shape as ``values``; the leading
    axis (bands) is preserved. Detailed spline prefilter construction is work
    in progress.
    """
    stencil = _BSPLINE_INT_STENCIL.get(order)
    if stencil is None:
        raise ValueError(f"Unsupported B-spline order {order!r} (use 3 or 5).")
    half = len(stencil) // 2
    c = np.array(values, dtype=np.complex128, copy=True)  # own buffer: the chain is in place
    # Spatial axes are the trailing three. An out-of-place chain allocates THREE
    # extra full-size cubes per axis (fft result, division result, ifft result)
    # while the previous ``c`` is still alive -- ~4 simultaneous copies, 108 GiB
    # at 501 bands and spline 150^3 (27 GiB per copy). Transforming and
    # dividing in place holds exactly one cube. pocketfft's in-place path runs the same algorithm, and
    # ``/=`` is elementwise, so the result is bit-identical.
    for axis in (-3, -2, -1):
        n = c.shape[axis]
        kernel = np.zeros(n, dtype=np.float64)
        for j, w in enumerate(stencil):
            kernel[(j - half) % n] = w
        kspec = _spfft.fft(kernel)  # 1D, small -- workers kwarg has no effect
        shape = [1] * c.ndim
        shape[axis] = n
        c = _spfft.fft(c, axis=axis, workers=PW_PGTO_N_WORKERS, overwrite_x=True)
        c /= kspec.reshape(shape)
        c = _spfft.ifft(c, axis=axis, workers=PW_PGTO_N_WORKERS, overwrite_x=True)
    return c


#: Simultaneous full-size cubes live inside one ``bspline_prefilter_periodic``
#: axis step (the working buffer plus pocketfft's in-place scratch and the
#: freshly written output). Used to convert the byte budget into a band chunk.
_PREFILTER_LIVE_COPIES = 4


def fit_pw_to_spline(
    wfc: QEWavefunction,
    spline_shape: tuple[int, int, int],
    order: int = 5,
    band_subset: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.complex128]:
    r"""Periodic B-spline coefficients on a uniform ``spline_shape``.

    Returns ``(n_band, *spline_shape)`` complex coefficients (``n_band =
    len(band_subset)`` when given, else every band). ``band_subset`` selects the
    bands before the transform, so discarded bands are never materialised; the
    result is bit-identical to slicing the full output with ``band_subset``.

    The band axis is chunked against ``PW_SPLINE_FIT_CHUNK_BUDGET_BYTES``: every
    band is independent, so live scratch is ``chunk`` bands instead of all of
    them, while the output table is written once. A budget covering all bands
    in one chunk degenerates to the single-shot form.
    """
    idx = None if band_subset is None else np.asarray(band_subset, dtype=np.int64)
    n_sel = int(wfc.coeffs.shape[0] if idx is None else idx.size)
    npts = int(np.prod(spline_shape))
    out = np.empty((n_sel, *spline_shape), dtype=np.complex128)
    bytes_per_band = _PREFILTER_LIVE_COPIES * 16 * npts
    chunk = max(1, min(n_sel, PW_SPLINE_FIT_CHUNK_BUDGET_BYTES // max(1, bytes_per_band)))
    for b0 in range(0, n_sel, chunk):
        sl = slice(b0, min(b0 + chunk, n_sel))
        sub = np.arange(n_sel, dtype=np.int64)[sl] if idx is None else idx[sl]
        psi = reconstruct_pw_on_grid(wfc, spline_shape, band_subset=sub)  # (npts, nb)
        nb = int(sub.size)
        cube = psi.T.reshape((nb, *spline_shape))
        out[sl] = bspline_prefilter_periodic(cube, order)
        del psi, cube
    return out


def evaluate_spline_orbitals(
    coefficients: npt.NDArray[np.complex128],
    eval_shape: tuple[int, int, int],
    order: int = 5,
    shift_frac: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> npt.NDArray[np.complex128]:
    r"""Evaluate spline orbitals on a uniform ``eval_shape`` mesh, ``(n_points, n_band)``.

    ``coefficients`` are ``(n_band, *spline_shape)`` from
    :func:`fit_pw_to_spline`. The evaluation grid point ``(i,j,k)`` maps to the
    spline-grid coordinate ``t_a = (i + shift_a)/eval_n_a * spline_n_a``.
    Detailed evaluation method is work in progress.
    """
    c = np.asarray(coefficients, dtype=np.complex128)
    n_band = c.shape[0]
    spline_shape = c.shape[1:]
    start = -((order - 1) // 2)  # -1 (cubic) / -2 (quintic)
    offsets = np.arange(start, start + order + 1)

    # Defensive band chunking: the first-axis gather materialises
    # ``(n_band, ne0 * (order+1), ns1, ns2)`` -- 302 GiB at
    # 501 bands, FFT 300^3, quintic -- while each band is independent.
    # Bound the largest per-chunk intermediate by the fidelity byte budget so
    # ANY caller is safe regardless of n_band; a budget covering all bands in
    # one chunk degenerates to a single-shot evaluation, and the
    # per-band arithmetic is unchanged either way (bit-identical outputs).
    bytes_per_band = 16 * max(
        int(eval_shape[0]) * (order + 1) * int(np.prod(spline_shape[1:])),  # axis-0 gather
        int(np.prod(eval_shape)),  # final (ne0, ne1, ne2) block
    )
    chunk = max(1, min(n_band, PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES // max(1, bytes_per_band)))
    out = np.empty((int(np.prod(eval_shape)), n_band), dtype=np.complex128)

    for b0 in range(0, n_band, chunk):
        cb = c[b0 : b0 + chunk]
        nb = cb.shape[0]
        # Contract one spatial axis per step. The freshly evaluated axis is
        # appended at the end so the next spline axis is always at position 1;
        # after three steps ``acc`` is (nb, ne0, ne1, ne2).
        acc = cb
        for axis in range(3):
            ns = spline_shape[axis]
            ne = eval_shape[axis]
            t = (np.arange(ne) + shift_frac[axis]) / ne * ns
            i = np.floor(t).astype(int)
            f = t - i
            w = _bspline_weights(f, order)  # (ne, order+1)
            idx = (i[:, None] + offsets[None, :]) % ns  # (ne, order+1)
            gathered = np.take(acc, idx.reshape(-1), axis=1)
            gathered = gathered.reshape((nb, ne, order + 1) + acc.shape[2:])
            contracted = np.einsum("nek...,ek->ne...", gathered, w)
            acc = np.moveaxis(contracted, 1, -1)
        out[:, b0 : b0 + nb] = acc.reshape(nb, -1).T
    return out


def project_pw_to_spline(
    reference: QEReference,
    mesh_factor: float = 0.75,
    order: int = 5,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
) -> ProjectionResult:
    r"""Convert the bands to a periodic uniform B-spline on the whole cell.

    A single spline (no domain partition) of grid
    ``round(mesh_factor * fft_grid)`` is fitted to every band. Fidelity is the
    interpolation error measured at the cell-midpoint mesh (half-step-shifted
    FFT grid), where the spline knots never coincide with the sample points, so
    even ``mesh_factor=1`` reports a genuine between-knot residual,

        r_n = || s_n - psi_n ||^2 / || psi_n ||^2   (on the shifted grid).

    A smaller ``mesh_factor`` coarsens the grid (the B-spline analogue of
    lowering the plane-wave cutoff), trading accuracy for memory. ``order`` is
    the spline degree (5 = quintic, C4, recommended for forces; 3 = cubic, C2).
    Detailed spline fitting method is work in progress.
    """
    fft_grid = reference.fft_grid
    spline_shape = tuple(max(order + 1, int(round(mesh_factor * n))) for n in fft_grid)

    # Nyquist-headroom check. The spline grid can represent
    # integer frequencies up to n_a // 2 per axis; the wavefunction carries
    # them up to max|miller_a|. Headroom = (n_a // 2) / max|miller_a|. At
    # headroom 1.0 the samples still determine the orbital (sampling theorem)
    # but a local B-spline cannot extract it: the VALUE error stays O(h^6)
    # while the LAPLACIAN error is O(h^4), so the build-time L2 residual looks
    # excellent while the determinant kinetic energy acquires a bias the
    # residual cannot see (measured at a 300 Ry cutoff: -9.5 mHa/twist at
    # headroom 1.0, -0.44 at 1.5, -0.06 at 2.0 = the QMCPACK-default grid).
    # 1.25 is the empirical edge of the safe region across Li/Si/H2O/O2.
    _millers_abs = np.max(np.abs(np.asarray(reference.wfc.millers, dtype=np.int64)), axis=0)
    _headroom = min((n // 2) / max(int(m), 1) for n, m in zip(spline_shape, _millers_abs))
    if _headroom < 1.25:
        logger.warning(
            "B-spline Nyquist headroom %.2f < 1.25 (grid %s vs max|G| index %s, "
            "mesh_factor %.3f): the spline can hold the orbital VALUES but not "
            "their Laplacian -- expect a kinetic-energy bias that the L2 "
            "residual below will NOT show. Converge the VMC energy and "
            "variance against mesh_factor (e.g. 0.75/1.0) before production; "
            "hard pseudopotentials at their minimum converged ecutwfc are the "
            "worst case.",
            _headroom,
            spline_shape,
            tuple(int(m) for m in _millers_abs),
            mesh_factor,
        )

    # Select the kept bands FIRST: the discarded (empty) bands are then never
    # transformed, and the full-table + occupied-copy pair (27 + 23 GiB at 501
    # bands) collapses to the single occupied table.
    band_idx = _select_band_indices(reference, occ_only, occ_tol, reference.wfc.n_bands)
    occ = reference.bands.occupations[band_idx]
    coeffs_b = fit_pw_to_spline(reference.wfc, spline_shape, order, band_subset=band_idx)
    # Unit-cell-normalized bands. The prefilter is linear, so scaling
    # the control points equals scaling the grid values; done in place to avoid
    # a second copy of the (possibly tens of GiB) table.
    cell_norm = _cell_normalization(reference)
    coeffs_b *= cell_norm

    # Fidelity check on the half-step-shifted FFT grid (knots avoided).
    # Band-chunked: the residual r_n is per-band independent, and an unchunked
    # evaluation holds all bands on the full grid at once -- three
    # ``(prod(fft_grid), n_band)`` complex arrays (~216 GiB each at 501 bands, FFT 300^3)
    # plus the 302 GiB internal gather -- for a diagnostic whose legitimate
    # product (the control-point table) is ~27 GiB. Peak follows
    # ``PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES``. The FFT / spline / subset
    # steps are bit-identical per band; only the final einsum reduction order
    # depends on the column count, so residuals agree with the single-shot
    # form to reduction round-off (~4e-14 relative measured).
    shift = (0.5, 0.5, 0.5)
    n_sel = int(band_idx.size)
    npts = int(np.prod(fft_grid))
    chunk = max(1, min(n_sel, PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES // max(1, 16 * npts)))
    num = np.empty(n_sel, dtype=np.float64)
    den = np.empty(n_sel, dtype=np.float64)
    for b0 in range(0, n_sel, chunk):
        sl = slice(b0, min(b0 + chunk, n_sel))
        spline_vals = evaluate_spline_orbitals(coeffs_b[sl], fft_grid, order, shift_frac=shift)
        psi_ref = reconstruct_pw_on_grid(reference.wfc, fft_grid, shift_frac=shift, band_subset=band_idx[sl])
        psi_ref *= cell_norm  # in place: no extra chunk-sized buffer
        diff = spline_vals - psi_ref
        num[sl] = np.real(np.einsum("pn,pn->n", diff.conj(), diff))
        den[sl] = np.real(np.einsum("pn,pn->n", psi_ref.conj(), psi_ref))
    residuals = num / den

    # Report the resulting control-point table size explicitly. It is
    # ``n_band * prod(spline_shape)``, so it scales as ``mesh_factor**3``: the
    # difference between mesh_factor 1.0 and 0.5 is a factor of 8 in both device
    # memory and the on-disk Hamiltonian. Printing the bytes next to the
    # residual makes that trade visible at conversion time rather than at the
    # first out-of-memory run.
    _table_bytes = int(coeffs_b.size) * int(coeffs_b.dtype.itemsize)
    logger.info(
        "PW->B-spline: order %d, grid %s from QE FFT grid %s (mesh_factor %.3f), %d bands, "
        "control-point table %.3f GiB (%s), mean residual %.3e, max %.3e.",
        order,
        spline_shape,
        tuple(fft_grid),
        mesh_factor,
        int(coeffs_b.shape[0]),
        _table_bytes / 1024**3,
        coeffs_b.dtype,
        float(np.sum(occ * residuals) / np.sum(occ)) if occ.sum() > 0 else float(np.mean(residuals)),
        float(np.max(residuals)) if residuals.size else float("nan"),
    )
    return ProjectionResult(
        residuals=residuals.astype(np.float64),
        band_indices=band_idx,
        occupations=occ,
        coefficients=np.empty((band_idx.size, 0), dtype=np.complex128),
        primitives=(),
        n_gto=0,
        n_pw=0,
        n_dropped=0,
        s_cond=1.0,
        s_cond_full=1.0,
        s_eig_min=1.0,
        # ``copy=False``: fit_pw_to_spline already returns complex128, and the
        # default ``astype`` copy duplicated the whole table (23 GiB at 501
        # bands) purely to assert a dtype it already had.
        spline_coefficients=coeffs_b.reshape(band_idx.size, -1).astype(np.complex128, copy=False),
        spline_shape=spline_shape,
        spline_order=order,
        mesh_factor=float(mesh_factor),
    )


# ---------------------------------------------------------------------------
# Additive pGTO + periodic B-spline conversion (PW -> pGTO + B-spline)
# ---------------------------------------------------------------------------
def project_pw_to_pgto_spline(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    mesh_factor: float = 1.0,
    order: int = 5,
    alpha_min: float = 0.0,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> ProjectionResult:
    r"""Convert the bands to an additive ``pGTO + periodic B-spline`` model.

    The orbital is represented as ``psi(r) ~ sum_mu c_mu g_mu(r) + s(r)`` where
    ``s`` is a single quintic (or cubic) B-spline over the whole cell.
    The quintic representation is C4 smooth. The
    ``alpha_min`` floor (drop primitives with exponent ``< alpha_min``) assigns
    the diffuse/long-range part to the spline. Mesh adequacy must be checked by
    energy/variance convergence; a small fit residual alone does not certify
    the mesh.

    Args:
        reference: Loaded QE reference (cell, fft_grid, wfc, bands).
        shells_per_atom: Per-atom fitted pGTO shells.
        mesh_factor: Scales the number of spline grid points per direction relative
            to the FFT grid (values below 1 request a coarser grid).
        order: Spline degree (5 = quintic / C4, recommended; 3 = cubic / C2).
        alpha_min: Exponent floor; diffuse primitives below it are dropped
            (``0`` keeps every primitive).
        occ_only: Project only occupied bands (else all bands).
        occ_tol: Occupation threshold defining 'occupied'.
        svd_rcond: Canonical-orthogonalisation eigenvalue floor for the pGTO fit.
        image_tol: AO-value truncation tolerance of the jQMC pGTO leaf (must
            match the value the leaf uses or the composite loses correctness).
            Real-space pGTO image sums use :func:`pgto_leaf_image_tolerances` of
            it, the same normalisation-aware envelope tolerance as the leaf.

    Returns:
        ProjectionResult carrying both ``coefficients``/``primitives`` (pGTO) and
        ``spline_coefficients``/``spline_shape``/``spline_order`` (the residual
        spline); ``residuals`` is the final combined fidelity.
    """
    cell = reference.cell
    fft_grid = reference.fft_grid
    spline_shape = tuple(max(order + 1, int(round(mesh_factor * n))) for n in fft_grid)

    # Nyquist-headroom check. The spline grid can represent
    # integer frequencies up to n_a // 2 per axis; the wavefunction carries
    # them up to max|miller_a|. Headroom = (n_a // 2) / max|miller_a|. At
    # headroom 1.0 the samples still determine the orbital (sampling theorem)
    # but a local B-spline cannot extract it: the VALUE error stays O(h^6)
    # while the LAPLACIAN error is O(h^4), so the build-time L2 residual looks
    # excellent while the determinant kinetic energy acquires a bias the
    # residual cannot see (measured at a 300 Ry cutoff: -9.5 mHa/twist at
    # headroom 1.0, -0.44 at 1.5, -0.06 at 2.0 = the QMCPACK-default grid).
    # 1.25 is the empirical edge of the safe region across Li/Si/H2O/O2.
    _millers_abs = np.max(np.abs(np.asarray(reference.wfc.millers, dtype=np.int64)), axis=0)
    _headroom = min((n // 2) / max(int(m), 1) for n, m in zip(spline_shape, _millers_abs))
    if _headroom < 1.25:
        logger.warning(
            "B-spline Nyquist headroom %.2f < 1.25 (grid %s vs max|G| index %s, "
            "mesh_factor %.3f): the spline can hold the orbital VALUES but not "
            "their Laplacian -- expect a kinetic-energy bias that the L2 "
            "residual below will NOT show. Converge the VMC energy and "
            "variance against mesh_factor (e.g. 0.75/1.0) before production; "
            "hard pseudopotentials at their minimum converged ecutwfc are the "
            "worst case.",
            _headroom,
            spline_shape,
            tuple(int(m) for m in _millers_abs),
            mesh_factor,
        )

    shells = filter_shells_by_alpha_min(shells_per_atom, alpha_min)
    prims = build_pgto_primitives(shells, reference.atom_positions)
    # Same per-primitive envelope tolerance as the emitted leaf (issue #120 / PR #122):
    # the bare ``image_tol`` dropped images the leaf keeps, leaving them un-subtracted.
    prim_image_tol = pgto_leaf_image_tolerances(prims, image_tol)

    band_idx = _select_band_indices(reference, occ_only, occ_tol, reference.wfc.n_bands)
    occ = reference.bands.occupations[band_idx]

    shift = (0.5, 0.5, 0.5)

    # --- Step 1: least-squares pGTO fit in PW space (no real grid). ---
    psi_local = _psi_pw_matrix(reference, band_idx)
    _, coeffs, n_dropped, s_cond, s_cond_full, s_eig_min = _project_bands_pw(
        reference, prims, psi_local, svd_rcond
    )  # coeffs (n_band, n_gto)

    # --- Step 2: residual in REAL space on the spline grid. ---
    # The downstream jQMC pgto leaf evaluates the full periodic Gaussian
    # image sum (not band-limited to the wfc G-set), so the residual fed
    # to the spline must subtract the same real-space pgto -- otherwise
    # the composite ``pgto_full + spline(psi - pgto_band_limited)`` leaves a
    # ``pgto_full - pgto_band_limited`` (high-G) residual that pollutes psi.
    #
    # Gauge (non-Gamma): everything on the grid lives in the cell-periodic
    # u-gauge. ``reconstruct_pw_on_grid`` returns ``sum_G c_G e^{iG.r}``
    # (no ``e^{ik.r}`` envelope) and the complex B-spline leaf multiplies
    # the envelope back at evaluation time -- so the pgto part must be
    # subtracted in the SAME gauge: ``e^{-ik.r} sum_T g e^{ik.T}``
    # (:func:`evaluate_pgto_expansion_u_gauge`). The u-gauge residual is
    # cell-periodic, which is also what ``bspline_prefilter_periodic``
    # assumes. Plain :func:`evaluate_pgto_expansion` here corrupts the
    # composite by O(1) at general k; at Gamma both forms are
    # byte-identical.
    k_cart_arr = np.asarray(reference.wfc.k_point, dtype=np.float64).reshape(3)
    grid_spl = grid_points(cell, spline_shape)
    # Band-chunked, and the band subset is applied BEFORE the IFFT: the
    # unchunked form built every band on the spline grid, sliced afterwards,
    # and held psi + pgto + residual + prefilter scratch simultaneously
    # (~5 full (n_band, *spline_shape) cubes; 27 GiB each at 501 bands). The
    # residual spline of one band depends on no other band, so the chunk loop
    # is a pure peak reduction -- same arithmetic, same result.
    _n_sel_s = int(band_idx.size)
    _npts_s = int(np.prod(spline_shape))
    _chunk_s = max(1, min(_n_sel_s, PW_SPLINE_FIT_CHUNK_BUDGET_BYTES // max(1, _PREFILTER_LIVE_COPIES * 16 * _npts_s)))
    spline_coeffs = np.empty((_n_sel_s, *spline_shape), dtype=np.complex128)
    # Unit-cell-normalized bands: the Step-1 pGTO coefficients come
    # from the cell-normalized right-hand side, so the grid bands are scaled by
    # the same Omega^{-1/2} (in place, no extra buffer) before the subtraction.
    cell_norm = _cell_normalization(reference)
    for _b0 in range(0, _n_sel_s, _chunk_s):
        _sl = slice(_b0, min(_b0 + _chunk_s, _n_sel_s))
        _nb = int(band_idx[_sl].size)
        psi_spl = reconstruct_pw_on_grid(reference.wfc, spline_shape, band_subset=band_idx[_sl])  # (n_pts, nb)
        psi_spl *= cell_norm
        pgto_spl = evaluate_pgto_expansion_u_gauge(cell, prims, coeffs[_sl], grid_spl, k_cart_arr, tol=prim_image_tol)
        psi_spl -= pgto_spl
        del pgto_spl
        spline_coeffs[_sl] = bspline_prefilter_periodic(psi_spl.T.reshape((_nb, *spline_shape)), order)
        del psi_spl

    # --- Fidelity of the combined model on the half-shifted FFT grid. ---
    # Use the same real-space (u-gauge) pgto evaluator as Step 2 so the
    # reported residual reflects what jQMC's composite will see at sample
    # points; the relative residual norm is gauge-invariant.
    # Band-chunked for the same reason as ``project_pw_to_spline``'s fidelity
    # block (per-band independent residual; unchunked form held every band on
    # the full shifted grid at once). Agrees with the single-shot form to
    # einsum reduction round-off (see the budget constant's comment).
    grid_shift = grid_points(cell, fft_grid, shift_frac=shift)
    n_sel = int(band_idx.size)
    npts_f = int(np.prod(fft_grid))
    chunk = max(1, min(n_sel, PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES // max(1, 16 * npts_f)))
    num = np.empty(n_sel, dtype=np.float64)
    den = np.empty(n_sel, dtype=np.float64)
    for b0 in range(0, n_sel, chunk):
        sl = slice(b0, min(b0 + chunk, n_sel))
        psi_ref = reconstruct_pw_on_grid(reference.wfc, fft_grid, shift_frac=shift, band_subset=band_idx[sl])
        psi_ref *= cell_norm
        pgto_shift = evaluate_pgto_expansion_u_gauge(cell, prims, coeffs[sl], grid_shift, k_cart_arr, tol=prim_image_tol)
        spline_shift = evaluate_spline_orbitals(spline_coeffs[sl], fft_grid, order, shift_frac=shift)
        diff = pgto_shift + spline_shift - psi_ref
        num[sl] = np.real(np.einsum("pn,pn->n", diff.conj(), diff))
        den[sl] = np.real(np.einsum("pn,pn->n", psi_ref.conj(), psi_ref))
    residuals = num / den

    logger.info(
        "PW->pGTO+B-spline (Fourier): %d pGTO (alpha_min %.3f, dropped %d) + spline order %d "
        "grid %s (mesh_factor %.3f), mean residual %.3e, max %.3e, cond(S) %.2e.",
        len(prims),
        alpha_min,
        n_dropped,
        order,
        spline_shape,
        mesh_factor,
        float(np.sum(occ * residuals) / np.sum(occ)) if occ.sum() > 0 else float(np.mean(residuals)),
        float(np.max(residuals)) if residuals.size else float("nan"),
        s_cond,
    )
    return ProjectionResult(
        residuals=residuals.astype(np.float64),
        band_indices=band_idx,
        occupations=occ,
        coefficients=coeffs,
        primitives=tuple(prims),
        n_gto=len(prims),
        n_pw=0,
        n_dropped=n_dropped,
        s_cond=s_cond,
        s_cond_full=s_cond_full,
        s_eig_min=s_eig_min,
        alpha_min=float(alpha_min),
        spline_coefficients=spline_coeffs.reshape(band_idx.size, -1).astype(np.complex128, copy=False),
        spline_shape=spline_shape,
        spline_order=order,
        mesh_factor=float(mesh_factor),
    )


def project_pw_to_pgto_spline_debug(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
    mesh_factor: float = 1.0,
    order: int = 5,
    alpha_min: float = 0.0,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = PW_PGTO_SVD_RCOND,
    image_tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> ProjectionResult:
    r"""Real-space reference twin of :func:`project_pw_to_pgto_spline` (numpy, no JAX).

    Retained only as the cross-check for the production path -- see the
    ``test_fourier_vs_real_*`` tests.
    """
    cell = reference.cell
    fft_grid = reference.fft_grid
    spline_shape = tuple(max(order + 1, int(round(mesh_factor * n))) for n in fft_grid)

    # Nyquist-headroom check. The spline grid can represent
    # integer frequencies up to n_a // 2 per axis; the wavefunction carries
    # them up to max|miller_a|. Headroom = (n_a // 2) / max|miller_a|. At
    # headroom 1.0 the samples still determine the orbital (sampling theorem)
    # but a local B-spline cannot extract it: the VALUE error stays O(h^6)
    # while the LAPLACIAN error is O(h^4), so the build-time L2 residual looks
    # excellent while the determinant kinetic energy acquires a bias the
    # residual cannot see (measured at a 300 Ry cutoff: -9.5 mHa/twist at
    # headroom 1.0, -0.44 at 1.5, -0.06 at 2.0 = the QMCPACK-default grid).
    # 1.25 is the empirical edge of the safe region across Li/Si/H2O/O2.
    _millers_abs = np.max(np.abs(np.asarray(reference.wfc.millers, dtype=np.int64)), axis=0)
    _headroom = min((n // 2) / max(int(m), 1) for n, m in zip(spline_shape, _millers_abs))
    if _headroom < 1.25:
        logger.warning(
            "B-spline Nyquist headroom %.2f < 1.25 (grid %s vs max|G| index %s, "
            "mesh_factor %.3f): the spline can hold the orbital VALUES but not "
            "their Laplacian -- expect a kinetic-energy bias that the L2 "
            "residual below will NOT show. Converge the VMC energy and "
            "variance against mesh_factor (e.g. 0.75/1.0) before production; "
            "hard pseudopotentials at their minimum converged ecutwfc are the "
            "worst case.",
            _headroom,
            spline_shape,
            tuple(int(m) for m in _millers_abs),
            mesh_factor,
        )

    shells = filter_shells_by_alpha_min(shells_per_atom, alpha_min)
    prims = build_pgto_primitives(shells, reference.atom_positions)
    prim_image_tol = pgto_leaf_image_tolerances(prims, image_tol)  # as the production path

    band_idx = _select_band_indices(reference, occ_only, occ_tol, reference.wfc.n_bands)
    occ = reference.bands.occupations[band_idx]

    shift = (0.5, 0.5, 0.5)

    # --- Step 1: least-squares pGTO fit on the dense FFT grid. ---
    # u-gauge at non-Gamma (see _u_gauge_design_matrix and the Step 2 note
    # in project_pw_to_pgto_spline): both the design matrix and the PW bands are
    # cell-periodic parts, so the fit coefficients match the Fourier fit
    # of the full Bloch orbital. Gamma path is byte-identical.
    k_cart_arr = np.asarray(reference.wfc.k_point, dtype=np.float64).reshape(3)
    grid_fft = grid_points(cell, fft_grid)
    dV = cell.volume / grid_fft.shape[0]
    phi_fft = _u_gauge_design_matrix(cell, prims, grid_fft, k_cart_arr, tol=prim_image_tol)
    # Unit-cell-normalized bands, matching the Fourier path.
    cell_norm = _cell_normalization(reference)
    psi_fft = reconstruct_pw_orbitals(reference.wfc, fft_grid)[:, band_idx] * cell_norm
    _, coeffs, n_dropped, s_cond, s_cond_full, s_eig_min = _project_bands(
        phi_fft, psi_fft, dV, svd_rcond
    )  # coeffs (n_band, n_gto)

    # --- Step 2: spline the smooth pGTO residual on the spline grid. ---
    grid_spl = grid_points(cell, spline_shape)
    psi_spl = reconstruct_pw_on_grid(reference.wfc, spline_shape)[:, band_idx] * cell_norm
    pgto_spl = evaluate_pgto_expansion_u_gauge(cell, prims, coeffs, grid_spl, k_cart_arr, tol=prim_image_tol)
    resid_spl = (psi_spl - pgto_spl).T.reshape((band_idx.size, *spline_shape))
    spline_coeffs = bspline_prefilter_periodic(resid_spl, order)

    # --- Fidelity of the combined model on the half-shifted FFT grid. ---
    grid_shift = grid_points(cell, fft_grid, shift_frac=shift)
    psi_ref = reconstruct_pw_on_grid(reference.wfc, fft_grid, shift_frac=shift)[:, band_idx] * cell_norm
    pgto_shift = evaluate_pgto_expansion_u_gauge(cell, prims, coeffs, grid_shift, k_cart_arr, tol=prim_image_tol)
    spline_shift = evaluate_spline_orbitals(spline_coeffs, fft_grid, order, shift_frac=shift)
    diff = pgto_shift + spline_shift - psi_ref
    num = np.real(np.einsum("pn,pn->n", diff.conj(), diff))
    den = np.real(np.einsum("pn,pn->n", psi_ref.conj(), psi_ref))
    residuals = num / den

    logger.info(
        "PW->pGTO+B-spline (real-space debug): %d pGTO (alpha_min %.3f, dropped %d) + spline order %d "
        "grid %s (mesh_factor %.3f), mean residual %.3e, max %.3e, cond(S) %.2e.",
        len(prims),
        alpha_min,
        n_dropped,
        order,
        spline_shape,
        mesh_factor,
        float(np.sum(occ * residuals) / np.sum(occ)) if occ.sum() > 0 else float(np.mean(residuals)),
        float(np.max(residuals)) if residuals.size else float("nan"),
        s_cond,
    )
    return ProjectionResult(
        residuals=residuals.astype(np.float64),
        band_indices=band_idx,
        occupations=occ,
        coefficients=coeffs,
        primitives=tuple(prims),
        n_gto=len(prims),
        n_pw=0,
        n_dropped=n_dropped,
        s_cond=s_cond,
        s_cond_full=s_cond_full,
        s_eig_min=s_eig_min,
        alpha_min=float(alpha_min),
        spline_coefficients=spline_coeffs.reshape(band_idx.size, -1).astype(np.complex128, copy=False),
        spline_shape=spline_shape,
        spline_order=order,
        mesh_factor=float(mesh_factor),
    )

r"""Variational optimisation of the uncontracted pGTO exponents.

The UPF path supplies an initial exponent ladder by fitting the sampled
``PP_PSWFC`` radial functions.  The basis used by the projector is, however,
all-primitive: every exponent is an independent Cartesian AO and the radial
fit coefficients are not contraction coefficients of the deployed basis.

This module adapts the constrained optimiser in trexio-tool to the jQMC
representation by optimising only ``z_p = log(alpha_p)``.  The objective is
the occupation-weighted spilling of the occupied QE orbitals, using the
module's plane-wave projection machinery.  A safety inequality constrains
the condition number of the raw full overlap of the basis that is emitted by
the converter.  Spilling remains the optimisation objective; conditioning is
not a competing cost.

Detailed method exposition is work in progress.
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

import dataclasses
from dataclasses import dataclass
from logging import getLogger
from math import comb, factorial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.optimize import NonlinearConstraint, minimize
from scipy.special import gammaincc, gammainccinv

from ._setting import (
    ATOL_BIT_EXACT,
    ATOL_PGTO_REFERENCE_POSITION_MATCH,
    PGTO_EXP_TAIL_CLIP_MARGIN,
    PGTO_EXP_TAIL_CLIP_SPREAD_RATIO,
    PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT,
    PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES,
    RTOL_BIT_EXACT,
    RTOL_PGTO_INITIAL_EXPONENT_MATCH,
    get_pgto_lattice_overlap_term_tolerance,
)
from .pw_pgto_projection import (
    _gamma_real_pw_basis_map,
    _generated_pgto_overlap_condition,
    build_pgto_primitives,
    fft_cube_millers,
    optimize_mixed_basis,
    project_pw_to_mixed,
    project_pw_to_pgto,
)
from .qe_wrapper import QEReference
from .structure import _primitive_image_list_indices
from .upf_wrapper import PrimitiveShell

jax.config.update("jax_enable_x64", True)

logger = getLogger("jqmc").getChild(__name__)


class PGTOExponentBoundsError(ValueError):
    r"""The exponent-tail ceiling cannot accommodate the requested seed/floor."""


class PGTOExponentOptimizationError(RuntimeError):
    r"""SLSQP did not report success, so its returned exponents are not used."""


@dataclass(frozen=True)
class PGTOExponentOptimizationResult:
    r"""Result and deployment diagnostics for exponent-only optimisation."""

    shells_per_atom: list[list[PrimitiveShell]]
    accepted: bool
    scipy_success: bool
    message: str
    n_iterations: int
    initial_spilling: float
    optimized_spilling: float
    initial_condition: float
    optimized_condition: float
    deployed_initial_spilling: float
    deployed_optimized_spilling: float
    deployed_initial_condition: float = float("nan")
    deployed_optimized_condition: float = float("nan")
    # Exponent-tail bound diagnostics.  ``exp_tail_eps`` and
    # ``g_max`` are None when the bound is disabled; ``alpha_report`` maps a
    # shell angular momentum l to {"max_alpha", "alpha_max", "ok"} of the
    # SLSQP point after the box clamp; ``n_clipped`` counts seed exponents
    # clipped to the bound before SLSQP.
    exp_tail_eps: float | None = None
    g_max: float | None = None
    alpha_report: dict[int, dict[str, float | bool]] = dataclasses.field(default_factory=dict)
    n_clipped: int = 0


@dataclass(frozen=True)
class _Layout:
    initial_exponents: npt.NDArray[np.float64]
    column_param_ids: npt.NDArray[np.int64]
    centers: npt.NDArray[np.float64]
    powers: npt.NDArray[np.int64]
    shell_param_ids: tuple[tuple[tuple[int, ...], ...], ...]
    # Shell angular momentum l of each unique exponent (n_unique,).
    uid_angular_l: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _ReferenceBlock:
    overlap_q_chunks: jax.Array
    overlap_mask_chunks: jax.Array
    subtract_q_chunks: jax.Array
    subtract_mask_chunks: jax.Array
    target_q_chunks: jax.Array
    target_psi_chunks: jax.Array
    target_mask_chunks: jax.Array
    occupations: jax.Array
    baseline_quality: jax.Array
    n_target_rows: int
    shared_q: bool
    has_pw: bool
    cell_volume: float
    n_pw: int
    condition_translations: jax.Array
    condition_phases: jax.Array
    condition_pw_source_indices: jax.Array
    condition_pw_source_weights: jax.Array
    condition_pw_diagonal_scale: jax.Array
    condition_pw_bulk_indices: jax.Array
    condition_pw_fixed_indices: jax.Array


def pw_tail_weight(l: int, alpha: npt.ArrayLike, g_max: float) -> npt.NDArray[np.float64]:
    r"""Fraction of a primitive's momentum-space L2 norm beyond ``|q| > g_max``.

    Parameters
    ----------
    l : int
        Nonnegative angular momentum of the primitive.
    alpha : float or array_like
        Positive Gaussian exponent in inverse length squared; vectorized over
        array inputs.
    g_max : float
        Positive momentum cutoff in inverse length.

    Returns
    -------
    float or ndarray
        Dimensionless tail fraction in ``[0, 1]``; the shape follows the
        broadcast shape of ``alpha``.  Values are monotonically increasing in
        ``alpha``.

    Detailed derivation is work in progress.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    return gammaincc(l + 1.5, g_max**2 / (2.0 * alpha))


def alpha_max_for_tail(l: int, g_max: float, eps: float) -> float:
    r"""Largest exponent whose tail weight beyond ``g_max`` does not exceed ``eps``.

    Parameters
    ----------
    l : int
        Nonnegative angular momentum of the primitive.
    g_max : float
        Positive momentum cutoff in inverse length.
    eps : float
        Tail-fraction tolerance in ``(0, 1)``.

    Returns
    -------
    float
        Scalar exponent in inverse length squared. Exponents greater than this
        value exceed the requested tail tolerance.

    Detailed derivation is work in progress.
    """
    x_min = float(gammainccinv(l + 1.5, eps))
    return float(g_max) ** 2 / (2.0 * x_min)


def _wfc_g_max(refs: Sequence[QEReference]) -> float:
    """Largest ``|G + k|`` of the WFC spheres of all references (1/Bohr).

    The q vectors are built exactly as in :func:`_reference_block`.  This is
    the WFC cutoff sphere, not the pGTO/PW handoff ``g_cut`` and not the FFT
    cube: the target orbitals carry no information outside it.
    """
    g2 = 0.0
    for ref in refs:
        q = np.asarray(ref.wfc.millers, dtype=np.float64) @ ref.cell.b
        k = np.asarray(ref.wfc.k_point, dtype=np.float64).reshape(3)
        if np.any(k != 0.0):
            q = q + k
        g2 = max(g2, float(np.max(np.sum(q * q, axis=1))))
    return float(np.sqrt(g2))


def _clip_seed_log_exponents(
    z0: npt.NDArray[np.float64],
    z_ub: npt.NDArray[np.float64],
    layout: "_Layout",
    floor_log: float | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Clip seed log-exponents above the exponent-tail bound.

    A single violating primitive of a shell is clipped to
    ``z_ub - PGTO_EXP_TAIL_CLIP_MARGIN``.  When two or more primitives of the
    same element shell violate the bound, clipping them all to that value
    would make the seed overlap exactly singular, so they are spread
    deterministically on the geometric ladder
    ``z_ub - PGTO_EXP_TAIL_CLIP_MARGIN - k ln(r)``, ``k = 0, 1, ...``,
    ``r = PGTO_EXP_TAIL_CLIP_SPREAD_RATIO``, in the order of their original
    exponents (largest on the top rung).  A rung closer than half a ladder
    step (``ln(r) / 2``) to an unclipped exponent of the same shell is
    skipped, and every used rung must stay strictly above ``floor_log``;
    otherwise ``ValueError`` is raised.  Returns the clipped ``z`` and the
    indices of the violating unique exponents.
    """
    violations = np.flatnonzero(z0 > z_ub)
    z = z0.copy()
    if violations.size == 0:
        return z, violations
    step = float(np.log(PGTO_EXP_TAIL_CLIP_SPREAD_RATIO))
    violating = set(int(i) for i in violations)
    seen: set[tuple[int, ...]] = set()
    for atom_map in layout.shell_param_ids:
        for uids in atom_map:
            if uids in seen:
                continue
            seen.add(uids)
            clipped = [uid for uid in uids if uid in violating]
            if not clipped:
                continue
            top = float(z_ub[clipped[0]]) - PGTO_EXP_TAIL_CLIP_MARGIN
            if len(clipped) == 1:
                z[clipped[0]] = top
                continue
            kept = [float(z0[uid]) for uid in uids if uid not in violating]
            order = sorted(clipped, key=lambda uid: (-float(z0[uid]), uid))
            rung = 0
            for uid in order:
                while True:
                    candidate = top - rung * step
                    rung += 1
                    if all(abs(candidate - other) >= 0.5 * step for other in kept):
                        break
                if floor_log is not None and not candidate > floor_log:
                    raise PGTOExponentBoundsError(
                        f"exp_tail_eps bound: {len(clipped)} primitives of one shell (l={int(layout.uid_angular_l[uid])}) "
                        f"exceed alpha_max={float(np.exp(z_ub[uid])):.6g}, and spreading them below it at ratio "
                        f"{PGTO_EXP_TAIL_CLIP_SPREAD_RATIO} reaches the exponent floor {float(np.exp(floor_log)):.6g}; "
                        "raise ecutwfc or exp_tail_eps, or lower alpha_min."
                    )
                z[uid] = candidate
    return z, violations


def _cartesian_components(angular_l: int) -> list[tuple[int, int, int]]:
    return [(nx, ny, angular_l - nx - ny) for nx in range(angular_l, -1, -1) for ny in range(angular_l - nx, -1, -1)]


def _build_layout(
    reference: QEReference,
    shells_per_atom: list[list[PrimitiveShell]],
) -> _Layout:
    """Build the element-tied exponent and AO-column maps.

    A unique variable is assigned to ``(element, shell ordinal, primitive
    slot)``.  Consequently equivalent atoms share a basis, while two channels
    of the same angular momentum remain distinct when they are distinct UPF
    shells.
    """
    if len(shells_per_atom) != len(reference.atom_positions):
        raise ValueError("shells_per_atom and atom_positions length mismatch.")

    labels = reference.atom_elements or reference.atom_labels
    if len(labels) != len(shells_per_atom):
        raise ValueError("QE atom labels/elements and shells_per_atom length mismatch.")

    key_to_uid: dict[tuple[str, int, int], int] = {}
    initial: list[float] = []
    uid_l: list[int] = []
    column_param_ids: list[int] = []
    centers: list[tuple[float, float, float]] = []
    powers: list[tuple[int, int, int]] = []
    shell_param_ids: list[tuple[tuple[int, ...], ...]] = []

    for atom_idx, (label, atom_shells, pos) in enumerate(zip(labels, shells_per_atom, reference.atom_positions, strict=True)):
        atom_map: list[tuple[int, ...]] = []
        for shell_idx, shell in enumerate(atom_shells):
            if len(shell.exponents) != len(shell.coefficients):
                raise ValueError(f"atom {atom_idx} shell {shell_idx}: exponent/coefficient length mismatch.")
            uids: list[int] = []
            for prim_idx, exponent in enumerate(shell.exponents):
                exponent = float(exponent)
                if not np.isfinite(exponent) or exponent <= 0.0:
                    raise ValueError(f"pGTO exponents must be finite and positive; got {exponent}.")
                key = (str(label), shell_idx, prim_idx)
                uid = key_to_uid.get(key)
                if uid is None:
                    uid = len(initial)
                    key_to_uid[key] = uid
                    initial.append(exponent)
                    uid_l.append(int(shell.angular_l))
                elif uid_l[uid] != int(shell.angular_l):
                    raise ValueError(
                        f"Equivalent atoms do not share the same shell angular momentum: key={key}, "
                        f"{uid_l[uid]} != {int(shell.angular_l)}."
                    )
                elif not np.isclose(initial[uid], exponent, rtol=RTOL_PGTO_INITIAL_EXPONENT_MATCH, atol=ATOL_BIT_EXACT):
                    raise ValueError(
                        f"Equivalent atoms do not share the same initial pGTO basis: key={key}, {initial[uid]} != {exponent}."
                    )
                uids.append(uid)
            atom_map.append(tuple(uids))
            center = (float(pos[0]), float(pos[1]), float(pos[2]))
            for power in _cartesian_components(int(shell.angular_l)):
                for uid in uids:
                    column_param_ids.append(uid)
                    centers.append(center)
                    powers.append(power)
        shell_param_ids.append(tuple(atom_map))

    return _Layout(
        initial_exponents=np.asarray(initial, dtype=np.float64),
        column_param_ids=np.asarray(column_param_ids, dtype=np.int64),
        centers=np.asarray(centers, dtype=np.float64),
        powers=np.asarray(powers, dtype=np.int64),
        shell_param_ids=tuple(shell_param_ids),
        uid_angular_l=np.asarray(uid_l, dtype=np.int64),
    )


def _reference_block(
    reference: QEReference,
    *,
    occ_only: bool,
    occ_tol: float,
    g_cut: float | None,
    fixed_mixed_overlap: bool,
    chunk_rows: int,
    condition_exponent_floor: float | None = None,
    condition_max_angular: int = 0,
) -> _ReferenceBlock:
    if occ_only:
        band_idx = reference.bands.occupied_band_indices(occ_tol)
    else:
        band_idx = np.arange(reference.wfc.n_bands, dtype=np.int64)
    if band_idx.size == 0:
        raise ValueError("Cannot optimise pGTO exponents without target bands.")

    wfc_millers = np.asarray(reference.wfc.millers, dtype=np.int64)
    g = wfc_millers.astype(np.float64) @ reference.cell.b
    q = g.copy()
    k = np.asarray(reference.wfc.k_point, dtype=np.float64).reshape(3)
    if np.any(k != 0.0):
        q = q + k

    # Rows are target orbitals, individually normalised in PW space.  This is
    # the convention used by trexio-tool's spilling objective.
    psi = np.asarray(reference.wfc.coeffs[band_idx, :], dtype=np.complex128)
    norms = np.sqrt(np.sum(np.abs(psi) ** 2, axis=1))
    if np.any(norms <= 0.0):
        raise ValueError("QE target contains a zero-norm band.")
    psi = psi / norms[:, None]
    occupations = np.asarray(reference.bands.occupations[band_idx], dtype=np.float64)
    if float(np.sum(occupations)) <= 0.0:
        occupations = np.ones_like(occupations)
    if g_cut is None:
        high_mask = np.ones(g.shape[0], dtype=bool)
        baseline_quality = np.zeros(psi.shape[0], dtype=np.float64)
        overlap_q = q
        target_q = q
        subtract_q = np.empty((0, 3), dtype=np.float64)
        shared_q = True
        has_pw = False
        n_pw = 0
    else:
        # Match the converter's retained-PW definition in
        # pw_gvectors_within: the cutoff is applied to the physical Bloch
        # wavevector |G+k|.  The optimizer must constrain exactly the basis
        # that the converter emits, including at non-Gamma k points.
        low_mask = np.linalg.norm(q, axis=1) <= float(g_cut)
        high_mask = ~low_mask
        # The retained low-G plane waves are exact orthogonal coordinate axes
        # in PW space, so their contribution to <psi|P|psi> is known without
        # any solve.  Only the complementary high-G rows enter the pGTO fit.
        baseline_quality = np.sum(np.abs(psi[:, low_mask]) ** 2, axis=1).real
        target_q = q[high_mask]
        if fixed_mixed_overlap:
            # ``project_pw_to_mixed`` forms S_gg on the complete FFT cube, then
            # Schur-complements the retained low-G PW axes.  The resulting
            # pGTO overlap is therefore a sum over every FFT-cube row except
            # the selected low-G Miller indices, while b still contains only
            # the band-limited wfc high-G rows.
            overlap_millers = fft_cube_millers(reference.fft_grid)
            overlap_q = overlap_millers.astype(np.float64) @ reference.cell.b
            if np.any(k != 0.0):
                overlap_q = overlap_q + k
            # The fixed projector's Schur complement removes the retained
            # low-G PW rows from the FFT-complete pGTO Gram matrix.
            subtract_q = q[low_mask]
            shared_q = False
        else:
            overlap_q = target_q
            subtract_q = q[low_mask]
            shared_q = True
        has_pw = True
        n_pw = int(np.count_nonzero(low_mask))

    def _chunks(
        array: np.ndarray,
        *,
        trailing_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_row = int(array.shape[0])
        n_chunk = (n_row + chunk_rows - 1) // chunk_rows
        if n_chunk == 0:
            return (
                np.empty((0, chunk_rows, *trailing_shape), dtype=array.dtype),
                np.empty((0, chunk_rows), dtype=np.float64),
            )
        padded = np.zeros((n_chunk * chunk_rows, *trailing_shape), dtype=array.dtype)
        padded[:n_row] = array
        mask = np.zeros(n_chunk * chunk_rows, dtype=np.float64)
        mask[:n_row] = 1.0
        return padded.reshape(n_chunk, chunk_rows, *trailing_shape), mask.reshape(n_chunk, chunk_rows)

    overlap_chunks, overlap_mask = _chunks(np.asarray(overlap_q), trailing_shape=(3,))
    subtract_chunks, subtract_mask = _chunks(np.asarray(subtract_q), trailing_shape=(3,))
    target_chunks, target_mask = _chunks(np.asarray(target_q), trailing_shape=(3,))
    target_psi = np.asarray(psi[:, high_mask].T, dtype=np.complex128)
    target_psi_chunks, _ = _chunks(target_psi, trailing_shape=(psi.shape[0],))

    if condition_exponent_floor is not None and condition_exponent_floor > 0.0:
        # For alpha,beta >= floor, rho=alpha*beta/(alpha+beta) >= floor/2.
        # The largest-angular, slowest-decaying pair therefore supplies one
        # fixed conservative lattice set for every point visited by SLSQP.
        indices = _primitive_image_list_indices(
            np.asarray(reference.cell.a, dtype=np.float64),
            0.5 * float(condition_exponent_floor),
            int(condition_max_angular),
            get_pgto_lattice_overlap_term_tolerance(np.float64),
        )
        translations = np.asarray(indices, dtype=np.float64) @ np.asarray(reference.cell.a, dtype=np.float64)
        k_frac = k @ np.linalg.inv(np.asarray(reference.cell.b, dtype=np.float64))
        phases = np.exp(2j * np.pi * (indices @ k_frac))
    else:
        translations = np.empty((0, 3), dtype=np.float64)
        phases = np.empty((0,), dtype=np.complex128)
    if has_pw and not np.any(k != 0.0):
        pw_source_indices, pw_source_weights, pw_diagonal_scale = _gamma_real_pw_basis_map(wfc_millers[low_mask])
    else:
        pw_source_indices = np.empty((0, 2), dtype=np.int32)
        pw_source_weights = np.empty((0, 2), dtype=np.complex128)
        pw_diagonal_scale = np.ones(n_pw, dtype=np.float64)
    return _ReferenceBlock(
        overlap_q_chunks=jnp.asarray(overlap_chunks, dtype=jnp.float64),
        overlap_mask_chunks=jnp.asarray(overlap_mask, dtype=jnp.float64),
        subtract_q_chunks=jnp.asarray(subtract_chunks, dtype=jnp.float64),
        subtract_mask_chunks=jnp.asarray(subtract_mask, dtype=jnp.float64),
        target_q_chunks=jnp.asarray(target_chunks, dtype=jnp.float64),
        target_psi_chunks=jnp.asarray(target_psi_chunks, dtype=jnp.complex128),
        target_mask_chunks=jnp.asarray(target_mask, dtype=jnp.float64),
        occupations=jnp.asarray(occupations, dtype=jnp.float64),
        baseline_quality=jnp.asarray(baseline_quality, dtype=jnp.float64),
        n_target_rows=int(target_q.shape[0]),
        shared_q=shared_q,
        has_pw=has_pw,
        cell_volume=float(reference.cell.volume),
        n_pw=n_pw,
        condition_translations=jnp.asarray(translations, dtype=jnp.float64),
        condition_phases=jnp.asarray(phases, dtype=jnp.complex128),
        condition_pw_source_indices=jnp.asarray(pw_source_indices, dtype=jnp.int32),
        condition_pw_source_weights=jnp.asarray(pw_source_weights, dtype=jnp.complex128),
        condition_pw_diagonal_scale=jnp.asarray(pw_diagonal_scale, dtype=jnp.float64),
        condition_pw_bulk_indices=jnp.asarray(np.flatnonzero(pw_diagonal_scale == 0.5), dtype=jnp.int32),
        condition_pw_fixed_indices=jnp.asarray(np.flatnonzero(pw_diagonal_scale != 0.5), dtype=jnp.int32),
    )


def _axis_poly_factor(n: int, q_axis: jax.Array, beta: jax.Array) -> jax.Array:
    """JAX twin of ``pw_pgto_projection._axis_poly_factor``."""
    if n == 0:
        return jnp.ones_like(q_axis, dtype=jnp.complex128) * jnp.ones_like(beta)
    if n == 1:
        return -1j * beta * q_axis
    if n == 2:
        return (beta - beta**2 * q_axis**2).astype(jnp.complex128)
    if n == 3:
        return -1j * (3.0 * beta**2 * q_axis - beta**3 * q_axis**3)
    if n == 4:
        return (3.0 * beta**2 - 6.0 * beta**3 * q_axis**2 + beta**4 * q_axis**4).astype(jnp.complex128)
    raise ValueError(f"Cartesian angular exponent n={n} exceeds the supported maximum (4).")


def _build_g(z: jax.Array, layout: _Layout, q: jax.Array) -> jax.Array:
    """Build the deployed uncontracted pGTO Fourier columns at Cartesian q rows."""
    unique_alpha = jnp.exp(z)
    param_ids = jnp.asarray(layout.column_param_ids)
    centers = jnp.asarray(layout.centers)
    powers = layout.powers
    q2 = jnp.sum(q * q, axis=1)
    n_pw = int(q.shape[0])
    n_col = int(layout.column_param_ids.size)
    out = jnp.zeros((n_pw, n_col), dtype=jnp.complex128)

    # Group equal Cartesian powers to keep the JAX graph proportional to the
    # number of angular components, not the number of atoms/primitives.
    for power_arr in np.unique(powers, axis=0):
        power = tuple(int(v) for v in power_arr)
        idx_np = np.nonzero(np.all(powers == power_arr[None, :], axis=1))[0]
        idx = jnp.asarray(idx_np, dtype=jnp.int32)
        alpha = unique_alpha[param_ids[idx]][:, None]
        beta = 1.0 / (2.0 * alpha)
        envelope = (jnp.pi / alpha) ** 1.5 * jnp.exp(-q2[None, :] / (4.0 * alpha))
        ax = _axis_poly_factor(power[0], q[None, :, 0], beta)
        ay = _axis_poly_factor(power[1], q[None, :, 1], beta)
        az = _axis_poly_factor(power[2], q[None, :, 2], beta)
        phase = jnp.exp(-1j * (centers[idx] @ q.T))
        # Keep the same raw primitive convention as build_gto_pw_matrix for
        # the spilling objective.  The generated-AO condition constraint
        # applies jQMC's intrinsic primitive normalisation separately.
        cols = (envelope * ax * ay * az * phase).T
        out = out.at[:, idx].set(cols)
    return out


def _hermitian_logcond(s: jax.Array) -> jax.Array:
    """Log condition number of the untruncated Hermitian overlap.

    The tiny floor merely keeps the safety constraint finite while SLSQP moves
    through an infeasible point; a non-positive deployed eigenvalue is still
    rejected by the exact deployed-overlap check.
    """
    eig = jnp.linalg.eigvalsh(0.5 * (s + s.conj().T))
    floor = jnp.asarray(1.0e-300, dtype=eig.dtype)
    return jnp.log(jnp.maximum(eig[-1], floor)) - jnp.log(jnp.maximum(eig[0], floor))


def _cart_primitive_normalisations(z: jax.Array, layout: _Layout) -> jax.Array:
    """Intrinsic normalisations of the Cartesian primitives emitted by jQMC."""
    alpha = jnp.exp(z)[jnp.asarray(layout.column_param_ids)]
    powers = np.asarray(layout.powers, dtype=np.int64)
    angular_l = jnp.asarray(np.sum(powers, axis=1), dtype=jnp.float64)
    factorial_ratio = np.asarray(
        [
            factorial(int(nx))
            * factorial(int(ny))
            * factorial(int(nz))
            / (factorial(2 * int(nx)) * factorial(2 * int(ny)) * factorial(2 * int(nz)))
            for nx, ny, nz in powers
        ],
        dtype=np.float64,
    )
    return jnp.sqrt(
        (2.0 * alpha / jnp.pi) ** 1.5 * (8.0 * alpha) ** angular_l * jnp.asarray(factorial_ratio, dtype=jnp.float64)
    )


def _periodic_pgto_overlap(z: jax.Array, layout: _Layout, block: _ReferenceBlock) -> jax.Array:
    """Differentiable raw overlap of the normalized periodic pGTO AOs."""
    translations = block.condition_translations
    phases = block.condition_phases
    n_image = int(translations.shape[0])
    if n_image == 0:
        raise ValueError("A positive exponent floor is required for the analytic periodic-overlap constraint.")

    centers = jnp.asarray(layout.centers, dtype=jnp.float64)
    powers_np = np.asarray(layout.powers, dtype=np.int64)
    if powers_np.size and int(np.max(powers_np)) > 4:
        raise ValueError("Cartesian angular exponent exceeds the supported maximum (4).")
    powers = jnp.asarray(powers_np, dtype=jnp.int32)
    alpha = jnp.exp(z)[jnp.asarray(layout.column_param_ids)]
    normalisations = _cart_primitive_normalisations(z, layout)
    n_col = int(layout.column_param_ids.size)

    # Keep the largest pair/image temporary near 32 MiB.  The row scan is
    # checkpointed, so reverse-mode differentiation recomputes one bounded
    # slab rather than retaining the complete n_col**2*n_image tensor.
    row_chunk = max(1, min(16, (32 * 1024**2) // max(8 * n_col * n_image * 8, 1)))
    n_chunk = (n_col + row_chunk - 1) // row_chunk
    padded_idx = np.minimum(np.arange(n_chunk * row_chunk, dtype=np.int32), n_col - 1)
    row_indices = jnp.asarray(padded_idx.reshape(n_chunk, row_chunk))
    row_masks = jnp.asarray(
        (np.arange(n_chunk * row_chunk) < n_col).reshape(n_chunk, row_chunk),
        dtype=jnp.float64,
    )
    binomial = jnp.asarray([[comb(n, i) if i <= n else 0 for i in range(5)] for n in range(5)], dtype=jnp.float64)

    def _axis_overlap(
        order_a: jax.Array,
        order_b: jax.Array,
        shift_a: jax.Array,
        shift_b: jax.Array,
        exponent_sum: jax.Array,
    ) -> jax.Array:
        value = jnp.zeros_like(shift_a)
        for i in range(5):
            coeff_i = binomial[order_a, i][:, None, None]
            power_a = jnp.maximum(order_a - i, 0)[:, None, None]
            term_a = coeff_i * shift_a**power_a
            for j in range(5):
                if (i + j) % 2 != 0:
                    continue
                coeff_j = binomial[order_b, j][None, :, None]
                power_b = jnp.maximum(order_b - j, 0)[None, :, None]
                k = (i + j) // 2
                double_factorial = 1 if k == 0 else int(np.prod(np.arange(1, 2 * k, 2)))
                moment = double_factorial * jnp.sqrt(jnp.pi) / (2.0**k * exponent_sum[:, :, None] ** (k + 0.5))
                value = value + term_a * coeff_j * shift_b**power_b * moment
        return value

    def _rows(_carry: None, inputs: tuple[jax.Array, jax.Array]) -> tuple[None, jax.Array]:
        mu, row_mask = inputs
        alpha_mu = alpha[mu]
        beta = alpha
        exponent_sum = alpha_mu[:, None] + beta[None, :]
        rho = alpha_mu[:, None] * beta[None, :] / exponent_sum
        delta = centers[None, :, None, :] + translations[None, None, :, :] - centers[mu, None, None, :]
        shift_a = beta[None, :, None, None] / exponent_sum[:, :, None, None] * delta
        shift_b = -alpha_mu[:, None, None, None] / exponent_sum[:, :, None, None] * delta
        primitive = jnp.exp(-rho[:, :, None] * jnp.sum(delta * delta, axis=3))
        for axis in range(3):
            primitive = primitive * _axis_overlap(
                powers[mu, axis],
                powers[:, axis],
                shift_a[:, :, :, axis],
                shift_b[:, :, :, axis],
                exponent_sum,
            )
        rows = normalisations[mu, None] * normalisations[None, :] * jnp.sum(primitive * phases[None, None, :], axis=2)
        return None, rows * row_mask[:, None]

    _, row_blocks = jax.lax.scan(jax.checkpoint(_rows), None, (row_indices, row_masks))
    overlap = row_blocks.reshape((-1, n_col))[:n_col, :]
    return 0.5 * (overlap + overlap.conj().T)


def _full_overlap_logcond(
    s_gg: jax.Array,
    g_low: jax.Array,
    normalisations: jax.Array,
    block: _ReferenceBlock,
) -> jax.Array:
    """Log condition number of the raw overlap of emitted pGTO+PW AOs."""
    volume = jnp.asarray(block.cell_volume, dtype=jnp.float64)
    # _build_g is the raw primitive transform.  Multiplication by the
    # intrinsic jQMC primitive normalisation gives the cross block of the
    # actual emitted AO basis.  No post-hoc rescaling of S is performed.
    normalized_g_low = g_low * normalisations[None, :]
    s_gp = normalized_g_low.conj().T
    is_gamma_real = block.condition_pw_source_indices.shape[0] > 0
    if is_gamma_real:
        s_gp = jnp.sum(
            s_gp[:, block.condition_pw_source_indices] * block.condition_pw_source_weights[None, :, :],
            axis=2,
        )
    # When the PW block is larger than the pGTO block, a thin QR of
    # S_gp^H rotates the PW basis into its row space plus an orthogonal
    # complement.  The complement contributes only the eigenvalue V, so this
    # is exactly the same raw full S spectrum without assembling an n_pw^2
    # matrix.
    if not is_gamma_real and g_low.shape[0] > s_gg.shape[0]:
        _, r = jnp.linalg.qr(normalized_g_low, mode="reduced")
        s_gp = r.conj().T
    gamma_reduced = is_gamma_real and block.condition_pw_bulk_indices.shape[0] > s_gg.shape[0]
    if gamma_reduced:
        # Gamma cos/sin columns all have norm V/2. Keep the exceptional
        # constant column (norm V) explicit, and rotate only this equal-norm
        # bulk. Its omitted orthogonal complement has eigenvalue V/2.
        bulk = s_gp[:, block.condition_pw_bulk_indices]
        q, _ = jnp.linalg.qr(jax.lax.stop_gradient(bulk.conj().T), mode="reduced")
        # Differentiate the projected original cross block, not QR. At this
        # point the bulk lies in Q's span; Hellmann--Feynman therefore gives
        # the same eigenvalue differential with Q held fixed. This also avoids
        # QR's inverse-R derivative for rank-deficient or zero cross blocks.
        s_gp = jnp.concatenate((bulk @ q, s_gp[:, block.condition_pw_fixed_indices]), axis=1)
        scales = jnp.concatenate(
            (jnp.full((s_gg.shape[0],), 0.5), block.condition_pw_diagonal_scale[block.condition_pw_fixed_indices])
        )
        s_pp = volume * jnp.diag(scales).astype(s_gg.dtype)
    elif is_gamma_real:
        s_pp = volume * jnp.diag(block.condition_pw_diagonal_scale).astype(s_gg.dtype)
    else:
        s_pp = volume * jnp.eye(s_gp.shape[1], dtype=s_gg.dtype)
    full_s = jnp.block(
        [
            [s_gg, s_gp],
            [s_gp.conj().T, s_pp],
        ]
    )
    eig = jnp.linalg.eigvalsh(0.5 * (full_s + full_s.conj().T))
    eig_min = eig[0]
    eig_max = eig[-1]
    if not is_gamma_real and normalized_g_low.shape[0] > s_gg.shape[0]:
        eig_min = jnp.minimum(eig_min, volume)
        eig_max = jnp.maximum(eig_max, volume)
    if gamma_reduced:
        eig_min = jnp.minimum(eig_min, volume / 2.0)
        eig_max = jnp.maximum(eig_max, volume / 2.0)
    floor = jnp.asarray(1.0e-300, dtype=eig.dtype)
    return jnp.log(jnp.maximum(eig_max, floor)) - jnp.log(jnp.maximum(eig_min, floor))


def _chunked_gram(
    z: jax.Array,
    layout: _Layout,
    q_chunks: jax.Array,
    mask_chunks: jax.Array,
) -> jax.Array:
    n_col = int(layout.column_param_ids.size)
    initial = jnp.zeros((n_col, n_col), dtype=jnp.complex128)

    def _body(
        carry: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        q_chunk, mask = inputs
        g = _build_g(z, layout, q_chunk) * mask[:, None]
        return carry + g.conj().T @ g, None

    gram, _ = jax.lax.scan(jax.checkpoint(_body), initial, (q_chunks, mask_chunks))
    return 0.5 * (gram + gram.conj().T)


def _chunked_rhs(
    z: jax.Array,
    layout: _Layout,
    q_chunks: jax.Array,
    psi_chunks: jax.Array,
    mask_chunks: jax.Array,
) -> jax.Array:
    n_col = int(layout.column_param_ids.size)
    n_band = int(psi_chunks.shape[2])
    initial = jnp.zeros((n_col, n_band), dtype=jnp.complex128)

    def _body(
        carry: jax.Array,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        q_chunk, psi_chunk, mask = inputs
        g = _build_g(z, layout, q_chunk) * mask[:, None]
        return carry + g.conj().T @ (psi_chunk * mask[:, None]), None

    rhs, _ = jax.lax.scan(
        jax.checkpoint(_body),
        initial,
        (q_chunks, psi_chunks, mask_chunks),
    )
    return rhs


def _chunked_gram_rhs(
    z: jax.Array,
    layout: _Layout,
    q_chunks: jax.Array,
    psi_chunks: jax.Array,
    mask_chunks: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    n_col = int(layout.column_param_ids.size)
    n_band = int(psi_chunks.shape[2])
    initial = (
        jnp.zeros((n_col, n_col), dtype=jnp.complex128),
        jnp.zeros((n_col, n_band), dtype=jnp.complex128),
    )

    def _body(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        gram, rhs = carry
        q_chunk, psi_chunk, mask = inputs
        g = _build_g(z, layout, q_chunk) * mask[:, None]
        gram = gram + g.conj().T @ g
        rhs = rhs + g.conj().T @ (psi_chunk * mask[:, None])
        return (gram, rhs), None

    (gram, rhs), _ = jax.lax.scan(
        jax.checkpoint(_body),
        initial,
        (q_chunks, psi_chunks, mask_chunks),
    )
    return 0.5 * (gram + gram.conj().T), rhs


def _spilling_and_logcond(
    z: jax.Array,
    layout: _Layout,
    blocks: tuple[_ReferenceBlock, ...],
    svd_rcond: float,
) -> tuple[jax.Array, jax.Array]:
    projected_weight = 0.0
    target_weight = 0.0
    worst_logcond = jnp.asarray(0.0, dtype=jnp.float64)
    for block in blocks:
        low_gram = None
        g_low = None
        if block.has_pw:
            low_gram = _chunked_gram(
                z,
                layout,
                block.subtract_q_chunks,
                block.subtract_mask_chunks,
            )
            q_low = block.subtract_q_chunks.reshape((-1, 3))[: block.n_pw]
            g_low = _build_g(z, layout, q_low)
        if block.shared_q:
            s, b = _chunked_gram_rhs(
                z,
                layout,
                block.overlap_q_chunks,
                block.target_psi_chunks,
                block.overlap_mask_chunks,
            )
        else:
            s = _chunked_gram(
                z,
                layout,
                block.overlap_q_chunks,
                block.overlap_mask_chunks,
            )
            if low_gram is not None:
                s = s - low_gram
            b = _chunked_rhs(
                z,
                layout,
                block.target_q_chunks,
                block.target_psi_chunks,
                block.target_mask_chunks,
            )
        if block.condition_translations.shape[0] > 0:
            condition_s_gg = _periodic_pgto_overlap(z, layout, block)
            if block.has_pw:
                logcond = _full_overlap_logcond(
                    condition_s_gg,
                    g_low,
                    _cart_primitive_normalisations(z, layout),
                    block,
                )
            else:
                logcond = _hermitian_logcond(condition_s_gg)
        elif block.has_pw:
            raise ValueError("Mixed pGTO+PW conditioning requires a positive exponent floor.")
        else:
            # Backwards-compatible fallback for direct pGTO-only use without
            # an exponent lower bound.  Final adoption is still decided by
            # the exact generated-AO overlap in _deployed_diagnostics.
            logcond = _hermitian_logcond(s)
        worst_logcond = jnp.maximum(worst_logcond, logcond)
        if block.n_target_rows == 0:
            quality = block.baseline_quality
            projected_weight = projected_weight + jnp.sum(block.occupations * quality)
            target_weight = target_weight + jnp.sum(block.occupations)
            continue
        # A differentiable canonical-orthogonalisation surrogate.  Directly
        # differentiating the truncated eigensystem is undefined at the exact
        # m-degeneracies of Cartesian shells (and produces NaN gradients).
        # A trace-scaled ridge has the same rcond scale, preserves those
        # degeneracies, and keeps the solve differentiable.  Candidate
        # exponents are still accepted only after the exact deployed projector
        # (hard eigenvalue truncation) is recomputed below.
        ridge = float(svd_rcond) * jnp.real(jnp.trace(s)) / s.shape[0]
        coeff = jnp.linalg.solve(s + ridge * jnp.eye(s.shape[0], dtype=s.dtype), b)
        quality = block.baseline_quality + jnp.real(jnp.sum(jnp.conj(b) * coeff, axis=0))
        projected_weight = projected_weight + jnp.sum(block.occupations * quality)
        target_weight = target_weight + jnp.sum(block.occupations)
    return 1.0 - projected_weight / target_weight, worst_logcond


def _replace_exponents(
    shells_per_atom: list[list[PrimitiveShell]],
    layout: _Layout,
    unique_exponents: npt.NDArray[np.float64],
) -> list[list[PrimitiveShell]]:
    out: list[list[PrimitiveShell]] = []
    for atom_shells, atom_map in zip(shells_per_atom, layout.shell_param_ids, strict=True):
        new_atom: list[PrimitiveShell] = []
        for shell, uids in zip(atom_shells, atom_map, strict=True):
            # Keep the PSWFC-fit coefficients paired only as inert metadata.
            # The projection remains uncontracted and never consumes them.
            pairs = sorted(
                ((float(unique_exponents[uid]), float(c)) for uid, c in zip(uids, shell.coefficients, strict=True)),
                key=lambda pair: pair[0],
            )
            exponents = tuple(pair[0] for pair in pairs)
            coefficients = tuple(pair[1] for pair in pairs)
            a = np.asarray(exponents, dtype=np.float64)
            overlap = (a[:, None] + a[None, :]) ** (-(int(shell.angular_l) + 1.5))
            new_atom.append(
                dataclasses.replace(
                    shell,
                    exponents=exponents,
                    coefficients=coefficients,
                    overlap_cond=float(np.linalg.cond(overlap)),
                )
            )
        out.append(new_atom)
    return out


def _deployed_diagnostics(
    references: Sequence[QEReference],
    shells_per_atom: list[list[PrimitiveShell]],
    *,
    occ_only: bool,
    occ_tol: float,
    svd_rcond: float,
    g_cut: float | None,
    alpha_min: float | None,
    fixed_mixed_overlap: bool,
) -> tuple[float, float]:
    weighted = 0.0
    weight_sum = 0.0
    worst_condition = 1.0
    for reference in references:
        if g_cut is None:
            result = project_pw_to_pgto(
                reference,
                shells_per_atom,
                occ_only=occ_only,
                occ_tol=occ_tol,
                svd_rcond=svd_rcond,
            )
        elif fixed_mixed_overlap:
            result = project_pw_to_mixed(
                reference,
                shells_per_atom,
                g_cut=float(g_cut),
                drop_diffuse=True,
                occ_only=occ_only,
                occ_tol=occ_tol,
                svd_rcond=svd_rcond,
                alpha_min=alpha_min,
            )
        else:
            sweep = optimize_mixed_basis(
                reference,
                shells_per_atom,
                residual_target=np.inf,
                g_cut_values=[float(g_cut)],
                drop_diffuse=True,
                fixed_alpha_min=alpha_min,
                occ_only=occ_only,
                occ_tol=occ_tol,
                svd_rcond=svd_rcond,
            )
            result = sweep.best
        occupations = result.occupations
        if float(np.sum(occupations)) <= 0.0:
            occupations = np.ones_like(occupations)
        weighted += float(np.sum(occupations * result.residuals))
        weight_sum += float(np.sum(occupations))
        worst_condition = max(worst_condition, float(result.s_cond_full))
    return weighted / weight_sum, worst_condition


def optimize_pgto_exponents(
    references: QEReference | Sequence[QEReference],
    shells_per_atom: list[list[PrimitiveShell]],
    *,
    occ_only: bool = True,
    occ_tol: float = 1.0e-6,
    svd_rcond: float = 1.0e-8,
    maxiter: int = 60,
    ftol: float = 1.0e-6,
    g_cut: float | None = None,
    exponent_floor: float | None = None,
    fixed_mixed_overlap: bool = False,
    deployed_fixed_mixed_overlap: bool | None = None,
    exp_tail_eps: float | None = None,
    g_max: float | None = None,
) -> PGTOExponentOptimizationResult:
    r"""Minimise pGTO spilling with respect to uncontracted exponents only.

    The returned basis is accepted only when it improves the spilling and the
    raw full overlap condition number recomputed through jQMC's deployed
    projector remains below ``1 / svd_rcond``.  A failed, non-improving, or
    ill-conditioned optimisation returns the original shells.

    ``fixed_mixed_overlap`` controls only whether the differentiable spilling
    objective uses the projector's FFT-complete or WFC-bounded representation.
    The condition constraint is the same analytic raw full overlap of the
    emitted AOs in either case.  ``deployed_fixed_mixed_overlap`` similarly
    selects the spilling audit used for adoption.

    ``exp_tail_eps`` limits each shell angular momentum ``l_p`` by
    ``alpha_max_for_tail(l_p, g_max, exp_tail_eps)``: no primitive may place more than
    the fraction ``exp_tail_eps`` of its momentum-space norm outside the WFC
    sphere ``|G + k| <= g_max``.  ``g_max`` defaults to the largest
    ``|G + k|`` over the WFC spheres of all ``references`` (every twist and both
    LSDA channels); it is never the pGTO/PW handoff ``g_cut``.  Seed exponents
    above the bound are clipped just below it with a warning, the spilling and
    deployed baselines are evaluated on the clipped seed, the SLSQP point is
    clamped to the box, and adoption additionally requires the bound to hold.
    A rejected point returns the clipped seed when one exists (see
    ``PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT``).  ``exp_tail_eps=None``
    (default) is bit-identical to the unbounded optimiser.

    Detailed method exposition is work in progress.
    """
    if isinstance(references, QEReference):
        refs = (references,)
    else:
        refs = tuple(references)
    if not refs:
        raise ValueError("At least one QE reference is required.")
    if svd_rcond <= 0.0:
        raise ValueError("svd_rcond must be positive.")

    layout = _build_layout(refs[0], shells_per_atom)
    # Exponent-tail box bound.  None leaves every quantity below
    # exactly as in the unbounded optimiser.
    floor_log = None if exponent_floor is None or float(exponent_floor) <= 0.0 else float(np.log(exponent_floor))
    z_ub: npt.NDArray[np.float64] | None = None
    tail_g_max: float | None = None
    if exp_tail_eps is not None:
        exp_tail_eps = float(exp_tail_eps)
        if not (0.0 < exp_tail_eps < 1.0):
            raise ValueError(f"exp_tail_eps must satisfy 0 < exp_tail_eps < 1; got {exp_tail_eps}.")
        tail_g_max = _wfc_g_max(refs) if g_max is None else float(g_max)
        if not (np.isfinite(tail_g_max) and tail_g_max > 0.0):
            raise ValueError(f"g_max must be finite and positive; got {tail_g_max}.")
        z_ub = np.asarray(
            [np.log(alpha_max_for_tail(int(l), tail_g_max, exp_tail_eps)) for l in layout.uid_angular_l],
            dtype=np.float64,
        )
        if floor_log is not None and np.any(z_ub <= floor_log):
            raise PGTOExponentBoundsError(
                f"exponent_floor={float(exponent_floor)} >= alpha_max(l, eps={exp_tail_eps}) at g_max={tail_g_max} "
                "for some shell l: the exponent box is infeasible; raise ecutwfc or exp_tail_eps, or lower alpha_min."
            )
    z0 = np.log(layout.initial_exponents)
    n_clipped = 0
    seed_shells = shells_per_atom
    if z_ub is not None:
        z_clipped, violations = _clip_seed_log_exponents(z0, z_ub, layout, floor_log)
        n_clipped = int(violations.size)
        if n_clipped:
            logger.warning(
                "exp_tail_eps bound: %d initial exponent(s) exceed alpha_max and were clipped: %s",
                n_clipped,
                ", ".join(
                    f"l={int(layout.uid_angular_l[i])}: {np.exp(z0[i]):.4g}->{np.exp(z_clipped[i]):.4g}" for i in violations[:8]
                ),
            )
            z0 = z_clipped
            # The clipped seed is the baseline of every "initial" diagnostic
            # and of the adoption test: the original seed violates the bound.
            seed_shells = _replace_exponents(shells_per_atom, layout, np.exp(z0))
    # Keep the autograd Fourier matrix below one quarter of the numpy
    # projection budget. Reverse-mode temporaries then remain bounded even
    # for 1024 pGTO columns; lax.scan reuses this fixed-size buffer.
    chunk_rows = min(
        8192,
        max(
            1,
            (PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES // 4) // max(16 * int(layout.column_param_ids.size), 1),
        ),
    )
    for ref in refs[1:]:
        if ref.atom_positions.shape != refs[0].atom_positions.shape or not np.allclose(
            ref.atom_positions, refs[0].atom_positions, rtol=RTOL_BIT_EXACT, atol=ATOL_PGTO_REFERENCE_POSITION_MATCH
        ):
            raise ValueError("All references must share atom positions for a common pGTO basis.")
    blocks = tuple(
        _reference_block(
            ref,
            occ_only=occ_only,
            occ_tol=occ_tol,
            g_cut=g_cut,
            fixed_mixed_overlap=fixed_mixed_overlap,
            chunk_rows=chunk_rows,
            condition_exponent_floor=exponent_floor,
            condition_max_angular=(2 * int(np.max(np.sum(layout.powers, axis=1))) if layout.powers.size else 0),
        )
        for ref in refs
    )
    cond_limit = 1.0 / float(svd_rcond)
    # As in trexio-tool, optimise just inside the true deployment boundary so
    # a finite-tolerance SLSQP result remains feasible after recomputation.
    log_cond_opt = float(np.log(cond_limit * (1.0 - 1.0e-3)))

    def _values(z: jax.Array) -> jax.Array:
        spilling, logcond = _spilling_and_logcond(z, layout, blocks, svd_rcond)
        return jnp.stack((spilling, log_cond_opt - logcond))

    def _values_and_jacobian(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        values, pullback = jax.vjp(_values, z)
        (jacobian,) = jax.vmap(pullback)(jnp.eye(2, dtype=jnp.float64))
        return values, jacobian

    values_and_jacobian = jax.jit(_values_and_jacobian)

    cache: dict[bytes, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

    def _eval(z: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        z_arr = np.asarray(z, dtype=np.float64)
        key = z_arr.tobytes()
        cached = cache.get(key)
        if cached is None:
            val_jax, jac_jax = values_and_jacobian(jnp.asarray(z_arr))
            val = np.asarray(val_jax, dtype=np.float64)
            jac = np.asarray(jac_jax, dtype=np.float64)
            cached = (val, jac)
            cache.clear()
            cache[key] = cached
        return cached

    # SLSQP's stopping and line-search tests use absolute objective changes.
    # pGTO spilling is commonly 1e-6--1e-4, so an equivalent constant scaling
    # prevents a scientifically useful gradient from being treated as zero.
    # The default SLSQP ftol is correspondingly 1e-6 on this scaled objective,
    # i.e. about 1e-10 in physical spilling; an ftol of 1e-10 would demand
    # ~1e-14 spilling changes and waste tens of iterations on large bases.
    objective_scale = 1.0e4
    fun = lambda z: objective_scale * float(_eval(z)[0][0])
    jac = lambda z: objective_scale * _eval(z)[1][0]
    exact_unbounded_pgto_condition = g_cut is None and (exponent_floor is None or float(exponent_floor) <= 0.0)
    exact_condition_cache: dict[bytes, float] = {}

    def _exact_pgto_logcondition(z: npt.ArrayLike) -> float:
        z_arr = np.asarray(z, dtype=np.float64)
        # A one-function overlap is a positive scalar, hence cond(S) == 1
        # independently of its exponent.  Avoid constructing a periodic image
        # list when SLSQP probes a very large trial step in that exact case.
        if layout.column_param_ids.size == 1:
            return 0.0
        key = z_arr.tobytes()
        cached = exact_condition_cache.get(key)
        if cached is None:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                exponents = np.exp(z_arr)
            if not np.all(np.isfinite(exponents)) or np.any(exponents <= 0.0):
                return float("inf")
            candidate_shells = _replace_exponents(shells_per_atom, layout, exponents)
            worst = 1.0
            for reference in refs:
                prims = build_pgto_primitives(candidate_shells, reference.atom_positions)
                worst = max(worst, _generated_pgto_overlap_condition(reference, prims))
            cached = float(np.log(worst))
            exact_condition_cache.clear()
            exact_condition_cache[key] = cached
        return cached

    if exact_unbounded_pgto_condition:

        def con_fun(z: npt.ArrayLike) -> npt.NDArray[np.float64]:
            logcondition = _exact_pgto_logcondition(z)
            value = log_cond_opt - logcondition if np.isfinite(logcondition) else -1.0e100
            return np.asarray([value], dtype=np.float64)

        def con_jac(z: npt.ArrayLike) -> npt.NDArray[np.float64]:
            z_arr = np.asarray(z, dtype=np.float64)
            step = 1.0e-5
            gradient = np.empty_like(z_arr)
            center = _exact_pgto_logcondition(z_arr)
            for index in range(z_arr.size):
                z_plus = z_arr.copy()
                z_minus = z_arr.copy()
                z_plus[index] += step
                z_minus[index] -= step
                plus = _exact_pgto_logcondition(z_plus)
                minus = _exact_pgto_logcondition(z_minus)
                if np.isfinite(plus) and np.isfinite(minus):
                    gradient[index] = -(plus - minus) / (2.0 * step)
                elif np.isfinite(plus) and np.isfinite(center):
                    gradient[index] = -(plus - center) / step
                elif np.isfinite(minus) and np.isfinite(center):
                    gradient[index] = -(center - minus) / step
                else:
                    gradient[index] = 0.0
            return gradient.reshape(1, -1)

    else:
        con_fun = lambda z: np.asarray([_eval(z)[0][1]], dtype=np.float64)
        con_jac = lambda z: _eval(z)[1][1].reshape(1, -1)
    constraints = [NonlinearConstraint(con_fun, 0.0, np.inf, jac=con_jac)]

    initial_values = _eval(z0)[0]
    initial_spilling = float(initial_values[0])
    initial_condition = (
        float(np.exp(_exact_pgto_logcondition(z0)))
        if exact_unbounded_pgto_condition
        else float(np.exp(log_cond_opt - initial_values[1]))
    )
    logger.info(
        "pGTO exponent optimisation: n_parameters=%d, n_columns=%d, n_references=%d; "
        "initial spilling=%.6e, worst full cond(S)=%.3e, limit=%.3e.",
        z0.size,
        layout.column_param_ids.size,
        len(refs),
        initial_spilling,
        initial_condition,
        cond_limit,
    )

    iteration_count = 0

    def _iteration_callback(z: npt.ArrayLike) -> None:
        nonlocal iteration_count
        iteration_count += 1
        spilling_now = float(_eval(z)[0][0])
        if iteration_count == 1 or iteration_count % 10 == 0:
            logger.info(
                "pGTO exponent optimisation iteration %d/%d: spilling=%.6e.",
                iteration_count,
                int(maxiter),
                spilling_now,
            )

    result = minimize(
        fun,
        z0,
        method="SLSQP",
        jac=jac,
        constraints=constraints,
        bounds=(
            (None if floor_log is None else [(floor_log, None)] * z0.size)
            if z_ub is None
            else [(floor_log, float(upper)) for upper in z_ub]
        ),
        callback=_iteration_callback,
        options={"ftol": float(ftol), "maxiter": int(maxiter), "disp": False},
    )
    # A point SLSQP does not report as a success is never used (issue 917):
    # scipy 1.16.0-1.16.1 stop with "Singular matrix E in LSQ subproblem" at a
    # log exponent of ~3.4e3, whose exponential overflows. That includes the
    # iteration limit: rerun with a larger ``maxiter`` instead.
    if not result.success:
        raise PGTOExponentOptimizationError(
            f"pGTO exponent optimisation failed: SLSQP status={int(result.status)}, "
            f"nit={int(getattr(result, 'nit', 0))}: {result.message}"
        )
    # Adopt the point returned by constrained SLSQP.  Feasibility and
    # spilling are independently recomputed with the production projector
    # below.  The only projection is the clamp onto the exponent-tail box
    # upper bound (SLSQP may exceed a box bound by ~1e-16); the result is
    # never otherwise projected or backtracked.
    z_opt = np.asarray(result.x, dtype=np.float64)
    if z_ub is not None:
        z_opt = np.minimum(z_opt, z_ub)
    final_values = _eval(z_opt)[0]
    optimized_spilling = float(final_values[0])
    optimized_condition = (
        float(np.exp(_exact_pgto_logcondition(z_opt)))
        if exact_unbounded_pgto_condition
        else float(np.exp(log_cond_opt - final_values[1]))
    )
    candidate = _replace_exponents(shells_per_atom, layout, np.exp(z_opt))

    deployed_fixed = fixed_mixed_overlap if deployed_fixed_mixed_overlap is None else deployed_fixed_mixed_overlap
    deployed_initial, deployed_initial_condition = _deployed_diagnostics(
        refs,
        seed_shells,
        occ_only=occ_only,
        occ_tol=occ_tol,
        svd_rcond=svd_rcond,
        g_cut=g_cut,
        alpha_min=exponent_floor,
        fixed_mixed_overlap=deployed_fixed,
    )
    deployed_optimized, deployed_optimized_condition = _deployed_diagnostics(
        refs,
        candidate,
        occ_only=occ_only,
        occ_tol=occ_tol,
        svd_rcond=svd_rcond,
        g_cut=g_cut,
        alpha_min=exponent_floor,
        fixed_mixed_overlap=deployed_fixed,
    )
    surrogate_feasible = bool(np.isfinite(optimized_condition) and optimized_condition <= cond_limit * (1.0 + 1.0e-8))
    deployed_feasible = bool(
        np.isfinite(deployed_optimized_condition) and deployed_optimized_condition <= cond_limit * (1.0 + 1.0e-8)
    )
    feasible = surrogate_feasible and deployed_feasible
    improved = bool(np.isfinite(deployed_optimized) and deployed_optimized < deployed_initial - 1.0e-12)
    alpha_report: dict[int, dict[str, float | bool]] = {}
    ok_exp_tail = True
    if z_ub is not None:
        for l in sorted({int(v) for v in layout.uid_angular_l}):
            mask = layout.uid_angular_l == l
            max_alpha = float(np.exp(np.max(z_opt[mask])))
            alpha_max = alpha_max_for_tail(l, tail_g_max, exp_tail_eps)
            # Audit in z = ln(alpha), the variable the box bounds: after the
            # clamp this is exact, whereas exp(ln(alpha_max)) may differ from
            # alpha_max by an ulp.
            ok = bool(np.max(z_opt[mask]) <= np.max(z_ub[mask]))
            alpha_report[l] = {"max_alpha": max_alpha, "alpha_max": alpha_max, "ok": ok}
            ok_exp_tail = ok_exp_tail and ok
        logger.info(
            "[exp_tail audit] eps=%.3e g_max=%.6f n_clipped=%d ok=%s; %s",
            exp_tail_eps,
            tail_g_max,
            n_clipped,
            ok_exp_tail,
            ", ".join(
                f"l={l}: max_alpha={entry['max_alpha']:.6g} / alpha_max={entry['alpha_max']:.6g}"
                for l, entry in alpha_report.items()
            ),
        )
    accepted = feasible and improved and ok_exp_tail
    rejected_shells = seed_shells if PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT else shells_per_atom
    selected_shells = candidate if accepted else rejected_shells
    logger.info(
        "pGTO exponent optimisation: scipy_success=%s, accepted=%s, nit=%d; "
        "autograd spilling %.6e -> %.6e, deployed spilling %.6e -> %.6e, "
        "search full cond(S)=%.3e, deployed full cond(S)=%.3e.",
        bool(result.success),
        accepted,
        int(getattr(result, "nit", 0)),
        initial_spilling,
        optimized_spilling,
        deployed_initial,
        deployed_optimized,
        optimized_condition,
        deployed_optimized_condition,
    )
    if not accepted:
        logger.warning(
            "pGTO exponent optimisation was not adopted (search_feasible=%s, deployed_feasible=%s, deployed_improved=%s, "
            "exp_tail_ok=%s): %s",
            surrogate_feasible,
            deployed_feasible,
            improved,
            ok_exp_tail,
            str(result.message),
        )

    return PGTOExponentOptimizationResult(
        shells_per_atom=selected_shells,
        accepted=accepted,
        scipy_success=bool(result.success),
        message=str(result.message),
        n_iterations=int(getattr(result, "nit", 0)),
        initial_spilling=initial_spilling,
        optimized_spilling=optimized_spilling,
        initial_condition=initial_condition,
        optimized_condition=optimized_condition,
        deployed_initial_spilling=deployed_initial,
        deployed_optimized_spilling=deployed_optimized,
        deployed_initial_condition=deployed_initial_condition,
        deployed_optimized_condition=deployed_optimized_condition,
        exp_tail_eps=exp_tail_eps,
        g_max=tail_g_max,
        alpha_report=alpha_report,
        n_clipped=n_clipped,
    )

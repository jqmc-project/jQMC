"""Assemble jQMC AO / MO / Geminal data from a QE projection result.

Bridges :mod:`jqmc.pw_pgto_projection` (which projects Quantum ESPRESSO PW bands
onto periodic basis sets and returns a :class:`ProjectionResult`) into the jQMC
AO/MO data structures used by the rest of the package
(:class:`Structure_data`, the AO leaves in :mod:`jqmc.atomic_orbital`,
:class:`MOs_data`, and :class:`Geminal_data`).

The single entry point is :func:`build_wavefunction_from_projection`. It
covers five conversion kinds matching the projection routines:

    * ``"pw"``          -> :class:`AOs_plane_wave_data` (Gamma-point real basis).
    * ``"bspline"``    -> :class:`AOs_bspline_data` (per-band whole-cell spline;
      one AO per band, identity MO matrix).
    * ``"pgto"``       -> :class:`AOs_cart_pgto_data` (all-primitive Cartesian
      pGTO, one primitive per AO).
    * ``"pgto+pw"``    -> :class:`AOs_composite_data` of pGTO + plane-wave.
    * ``"pgto+bspline"`` -> :class:`AOs_composite_data` of pGTO + per-band
      spline (the spline holds the smooth residual ``psi - pGTO``).

MO normalization convention: every kind emits MOs in
the standard unit-cell-normalized convention ``integral_Omega |psi_n|^2 = 1``
(QE's ``psi_n = Omega^{-1/2} sum_G c_{n,G} e^{i(G+k).r}``; the ``Omega^{-1/2}``
is carried by the MO / spline coefficients, the basis functions are
unchanged). Pure ``"pgto"`` MOs are additionally orthonormalized in the true
periodic AO metric by a band-order Cholesky step (on by default,
``orthonormalize=False`` disables it), so ``conj(C) S_AO C^T = I`` and the
span of every lowest-N band subset is unchanged. ``"pw"`` / ``"bspline"`` /
``"pgto+pw"`` / ``"pgto+bspline"`` MOs are normalized to ``1 - r_n`` (``r_n``
the projection residual) and keep their natural, small non-orthogonality; the
mixed kinds are never orthonormalized (their metric needs the PW / spline
blocks and the cross overlap, and orthonormalizing only the pGTO part would
break the pGTO / PW ratio). HDF5 files written before this change carry a
common ``sqrt(Omega)`` MO factor and remain valid (single-determinant physics
is unchanged).

Precision Zones:
    This module owns no Precision Zone; it computes in NumPy fp64.

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

from collections.abc import Sequence
from dataclasses import replace
from logging import getLogger

import os

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.special

from ._setting import (
    ATOL_PGTO_PRIMITIVE_CENTER_MATCH,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_TILING_INTEGER_RESIDUE,
    PW_PGTO_SVD_RCOND,
    RTOL_BIT_EXACT,
)
from .atomic_orbital import (
    AOs_base_data,
    AOs_bspline_complex_data,
    AOs_bspline_data,
    AOs_bspline_tiled_complex_data,
    AOs_cart_pgto_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
    build_AOs_cart_pgto_complex_data,
    compute_overlap_matrix,
)
from .coulomb_potential import Coulomb_potential_data, build_ewald_data
from .determinant import Geminal_data, SD_data
from .hamiltonians import Hamiltonian_data
from .jastrow_factor import (
    DTN_DEFAULT_CUTOFF_RATIO,
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from .molecular_orbital import MOs_data
from .pw_pgto_projection import CartPrimitive, ProjectionResult
from .qe_wrapper import QEReference
from .structure import Structure_data, normalize_tiling_matrix, tile_structure_data
from .upf_wrapper import ECPData, UPFExponentInit
from .wavefunction import Wavefunction_data

logger = getLogger("jqmc").getChild(__name__)


_SUPPORTED_KINDS = ("pw", "bspline", "pgto", "pgto+pw", "pgto+bspline")

_DTN_PRESET_MAP: dict[str, tuple[bool, bool, bool]] = {
    "u": (True, False, False),
    "u_chi": (True, True, False),
    "u_f": (True, False, True),
    "u_chi_f": (True, True, True),
}


def build_jastrow_dtn_from_options(
    structure_data: Structure_data,
    *,
    jastrow_dtn: str | None,
    coulomb_potential_data: Coulomb_potential_data,
    dtn_c: int = 4,
    dtn_n_u: int = 8,
    dtn_n_chi: int = 8,
    dtn_n_eN: int = 4,
    dtn_n_ee: int = 4,
    dtn_cutoff_ratio_u: float = DTN_DEFAULT_CUTOFF_RATIO,
    dtn_cutoff_ratio_chi: float = DTN_DEFAULT_CUTOFF_RATIO,
    dtn_cutoff_ratio_f: float = DTN_DEFAULT_CUTOFF_RATIO,
) -> Jastrow_DTN_data | None:
    """Build the shared DTN Jastrow used by single- and multi-twist QE conversion."""
    if jastrow_dtn is None:
        return None
    choice = str(jastrow_dtn).lower()
    if choice == "none":
        return None
    if choice not in _DTN_PRESET_MAP:
        raise ValueError(f"jastrow_dtn={jastrow_dtn!r} is not recognised; allowed: none, u, u_chi, u_f, u_chi_f.")
    use_u, use_chi, use_f = _DTN_PRESET_MAP[choice]

    atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.int64)
    unique_Z = np.unique(atomic_numbers)
    effective_charge_per_atom = np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)
    pseudized_per_atom = np.asarray(coulomb_potential_data.pseudized_per_atom, dtype=np.bool_)
    if effective_charge_per_atom.shape != atomic_numbers.shape or pseudized_per_atom.shape != atomic_numbers.shape:
        raise ValueError("DTN and Coulomb data must describe the same number of atoms.")
    z_eff_per_species: list[float] = []
    pseudized_per_species: list[bool] = []
    for atomic_number in unique_Z:
        atoms_of_species = atomic_numbers == atomic_number
        species_effective_charges = np.unique(effective_charge_per_atom[atoms_of_species])
        species_pseudized_flags = np.unique(pseudized_per_atom[atoms_of_species])
        if species_effective_charges.size != 1 or species_pseudized_flags.size != 1:
            raise ValueError(
                f"Atoms of element Z={int(atomic_number)} do not share one ECP/effective-charge policy; "
                "assign distinct DTN species before building the Hamiltonian."
            )
        z_eff_per_species.append(float(species_effective_charges[0]))
        pseudized_per_species.append(bool(species_pseudized_flags[0]))

    return Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=tuple(z_eff_per_species),
        pseudized_per_species=tuple(pseudized_per_species),
        use_u=use_u,
        use_chi=use_chi,
        use_f=use_f,
        cutoff_smoothness=int(dtn_c),
        n_order_u=int(dtn_n_u),
        n_order_chi=int(dtn_n_chi),
        n_order_eN=int(dtn_n_eN),
        n_order_ee=int(dtn_n_ee),
        # Periodic cells use the comoving ratio policy (L = ratio * R_ins); molecular
        # systems take the absolute geometry-derived defaults and reject ratios.
        cutoff_ratio_u=float(dtn_cutoff_ratio_u) if structure_data.pbc_flag else None,
        cutoff_ratio_chi=float(dtn_cutoff_ratio_chi) if structure_data.pbc_flag else None,
        cutoff_ratio_f=float(dtn_cutoff_ratio_f) if structure_data.pbc_flag else None,
    )


# ---------------------------------------------------------------------------
# Structure construction
# ---------------------------------------------------------------------------
def _build_structure_data(
    qe_reference: QEReference,
) -> tuple[Structure_data, tuple[float, float, float] | None]:
    """Map a :class:`QEReference` cell + atoms to a periodic :class:`Structure_data`.

    QE stores lengths in Bohr (atomic units), which is also the jqmc convention,
    so the cell rows are copied verbatim. Element symbols come from each atom's
    UPF-resolved element (``qe_reference.atom_elements``), kept distinct from the
    raw QE species labels ('O' vs 'O1'); atomic numbers are mapped from the
    elements via the same table used by :mod:`jqmc.trexio_wrapper`.

    Returns ``(structure_data, k_point_frac_or_None)``: the structure_data
    has no notion of k (k lives on the AO leaf, not the structure); the
    second element is the fractional twist that the downstream AO-leaf
    builders should attach to the complex Bloch leaves. ``None`` selects
    Gamma, i.e. the real-path dispatch.
    """
    cell = qe_reference.cell
    structure_data = Structure_data.from_qe_reference(qe_reference)
    # Convert QE's Cartesian ``xk`` (1/bohr) to fractional reciprocal-
    # lattice coordinates: ``xk = k_frac @ cell.b``, so
    # ``k_frac = xk @ inv(cell.b)``. ``cell.b`` rows are the reciprocal
    # vectors in 1/bohr (with the 2pi convention,
    # ``cell.a @ cell.b.T == 2pi I``).
    xk_cart = np.asarray(qe_reference.wfc.k_point, dtype=np.float64).reshape(3)
    if np.any(xk_cart != 0.0):
        k_frac = xk_cart @ np.linalg.inv(cell.b)
        k_point_frac: tuple[float, float, float] | None = (float(k_frac[0]), float(k_frac[1]), float(k_frac[2]))
        logger.info(
            "QE -> jQMC: non-Gamma twist loaded, xk (1/bohr Cartesian) = %s, k_point (fractional reciprocal-lattice) = %s",
            xk_cart.tolist(),
            k_point_frac,
        )
    else:
        k_point_frac = None
    return structure_data, k_point_frac


# ---------------------------------------------------------------------------
# Electron / occupation bookkeeping
# ---------------------------------------------------------------------------
#: QMC fixes the electron numbers, so the QE -> jQMC conversion accepts only
#: integer fillings: every band occupation must be 0 or 1 within this
#: tolerance. Fractional occupations (metallic smearing, non-integer
#: tot_magnetization) cannot be represented by a Slater determinant --
#: silently rounding them would build a wavefunction that does not
#: correspond to the DFT state.
OCC_BINARY_TOL = 1.0e-6


def check_binary_occupations(
    occupations: npt.NDArray[np.float64],
    *,
    channel: str = "",
    tol: float = OCC_BINARY_TOL,
) -> None:
    """Raise ``ValueError`` unless every band occupation is 0 or 1 within ``tol``.

    ``channel`` is a short label ("up" / "dn" / "" for restricted) used in the
    error message. A tiny negative occupation (Methfessel-Paxton smearing on a
    gapped state can produce ~-1e-26) counts as 0.
    """
    occ = np.asarray(occupations, dtype=np.float64)
    if occ.size == 0:
        return
    dist = np.minimum(np.abs(occ), np.abs(occ - 1.0))
    if float(dist.max()) > tol:
        bad = int(np.argmax(dist))
        chan = f"{channel}-channel " if channel else ""
        raise ValueError(
            f"QE -> jQMC: {chan}occupation of band {bad} is {occ[bad]:.6f}, "
            f"neither 0 nor 1 (tol {tol:.0e}). QMC fixes the electron numbers, "
            "so only integer fillings with binary band occupations are accepted "
            "(no grand-canonical TABC yet). Re-run QE with occupations='fixed' "
            "(and, for nspin=2, an integer tot_magnetization); do not use smearing."
        )


# ---------------------------------------------------------------------------
# Plane-wave AO leaf assembly
# ---------------------------------------------------------------------------
def _pw_real_basis_from_millers(
    millers: npt.NDArray[np.int64],
    cell_b: npt.NDArray[np.float64],
    pw_coefficients: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map a Hermitian PW set ``{c_G}`` to the real cos/sin basis + MO coeffs.

    For a Gamma-point real orbital ``c_{-G} = conj(c_G)``, so the contribution
    of a Hermitian ``(G, -G)`` pair is ``2 Re(c_G) cos(G.r) - 2 Im(c_G) sin(G.r)``;
    the ``G = 0`` term contributes the real constant ``c_0``. Pairs are deduped
    by keeping only one representative per ``(G, -G)`` partner (the canonical
    lexicographically-greater Miller index).

    Returns ``(g_vectors, phase_is_sin, real_coeffs)`` where:
      * ``g_vectors`` ``(num_ao, 3)`` Cartesian, one row per real basis fn,
      * ``phase_is_sin`` ``(num_ao,)`` 1 -> sin, 0 -> cos / constant,
      * ``real_coeffs`` ``(n_band, num_ao)`` the per-band real coefficients.
    """
    millers = np.asarray(millers, dtype=np.int64)
    g_all = millers.astype(np.float64) @ cell_b  # (n_g, 3) Cartesian
    n_band = int(pw_coefficients.shape[0])

    # Index map for lookup of -G's partner.
    index_of: dict[tuple[int, int, int], int] = {tuple(m.tolist()): i for i, m in enumerate(millers)}

    visited = np.zeros(millers.shape[0], dtype=bool)
    g_rows: list[np.ndarray] = []
    sin_flags: list[float] = []
    coeff_cols: list[np.ndarray] = []

    for i, m in enumerate(millers):
        if visited[i]:
            continue
        visited[i] = True
        c_i = pw_coefficients[:, i]
        if np.all(m == 0):
            # Constant term: cos(0) = 1, real coefficient is Re(c_0).
            g_rows.append(g_all[i])
            sin_flags.append(0.0)
            coeff_cols.append(np.real(c_i).astype(np.float64))
            continue
        neg = tuple((-m).tolist())
        j = index_of.get(neg, -1)
        if j >= 0 and not visited[j]:
            # Canonical partner: pick lex-greater Miller index as representative.
            mi = tuple(m.tolist())
            if mi < neg:
                # Use the other partner as canonical so it does not get re-emitted.
                visited[j] = True
                c_repr = pw_coefficients[:, j]
                g_repr = g_all[j]
            else:
                visited[j] = True
                c_repr = c_i
                g_repr = g_all[i]
            # psi += 2 Re(c) cos(G.r) - 2 Im(c) sin(G.r) using the representative.
            g_rows.append(g_repr)
            sin_flags.append(0.0)
            coeff_cols.append((2.0 * np.real(c_repr)).astype(np.float64))
            g_rows.append(g_repr)
            sin_flags.append(1.0)
            coeff_cols.append((-2.0 * np.imag(c_repr)).astype(np.float64))
        else:
            # Lone G with no -G partner in the set (should not happen for a
            # gamma-expanded wfc, but keep a real-only fallback): treat as cos.
            g_rows.append(g_all[i])
            sin_flags.append(0.0)
            coeff_cols.append(np.real(c_i).astype(np.float64))

    g_vectors = np.asarray(g_rows, dtype=np.float64)
    phase_is_sin = np.asarray(sin_flags, dtype=np.float64)
    # coeff_cols holds n_ao arrays each of length n_band; stack into (n_band, n_ao).
    real_coeffs = np.stack(coeff_cols, axis=1) if coeff_cols else np.zeros((n_band, 0), dtype=np.float64)
    return g_vectors, phase_is_sin, real_coeffs


def _build_pw_leaf_and_mo(
    qe_reference: QEReference,
    projection_result: ProjectionResult,
    structure_data: Structure_data,
    k_point_frac: tuple[float, float, float] | None,
) -> tuple[AOs_base_data, npt.NDArray]:
    """Build a plane-wave AO leaf + per-band MO coefficients.

    Dispatches on ``k_point_frac``:

    * Gamma (``None``): real cos/sin basis (each Hermitian ``(G, -G)`` pair
      contributes 2 real basis functions
      ``2 Re(c_G) cos(G.r) - 2 Im(c_G) sin(G.r)``); returns
      :class:`AOs_plane_wave_data` and a ``(n_band, num_ao)`` float64
      coefficient block.
    * Non-Gamma (set): complex ``e^{i (G+k) . r}`` basis with one AO per
      stored ``G`` and the projection's ``pw_coefficients`` as the per-band
      complex coefficients; returns :class:`AOs_plane_wave_complex_data`
      and a ``(n_band, num_ao)`` complex128 coefficient block.

    ``projection_result.pw_coefficients`` already carry the unit-cell
    normalization ``Omega^{-1/2} c_G`` (applied once in
    :mod:`jqmc.pw_pgto_projection` for both ``"pw"`` and the PW block of
    ``"pgto+pw"``), so no further scale is applied here.
    """
    if projection_result.pw_coefficients.size == 0 or projection_result.pw_millers.size == 0:
        raise ValueError(
            "ProjectionResult has no PW block; expected kind='pw' / 'pgto+pw' from project_pw_to_pw or project_pw_to_mixed."
        )
    if k_point_frac is None:
        cell_b = qe_reference.cell.b
        g_vectors, phase_is_sin, real_coeffs = _pw_real_basis_from_millers(
            projection_result.pw_millers,
            cell_b,
            projection_result.pw_coefficients,
        )
        aos = AOs_plane_wave_data.build(structure_data, g_vectors, phase_is_sin)
        return aos, real_coeffs
    # Non-Gamma: complex e^{i(G+k).r} basis. No Hermitian pairing -- each
    # G carries an independent complex coefficient c_G (QE stores the
    # half-shell at non-Gamma; the c_{-G} = conj(c_G) symmetry only holds
    # at Gamma).
    millers = np.asarray(projection_result.pw_millers, dtype=np.int64)
    cell_b = qe_reference.cell.b
    g_vectors = millers.astype(np.float64) @ cell_b  # (n_g, 3) Cartesian
    aos = AOs_plane_wave_complex_data.build(structure_data, g_vectors, k_point=k_point_frac)
    complex_coeffs = np.asarray(projection_result.pw_coefficients, dtype=np.complex128)
    return aos, complex_coeffs


# ---------------------------------------------------------------------------
# B-spline AO leaf assembly
# ---------------------------------------------------------------------------
def _build_bspline_leaf_and_mo(
    projection_result: ProjectionResult,
    structure_data: Structure_data,
    k_point_frac: tuple[float, float, float] | None,
) -> tuple[AOs_base_data, npt.NDArray]:
    """Build a per-band B-spline AO leaf + identity MO matrix.

    Each AO is one whole-cell periodic spline reproducing one band, so the
    MO matrix is the ``n_band x n_band`` identity. Dispatches on ``k_point_frac``:

    * Gamma (``None``): real :class:`AOs_bspline_data` + ``float64`` identity MO.
    * Non-Gamma: complex :class:`AOs_bspline_complex_data` (complex u_k
      control points + per-electron ``e^{i k . r}`` Bloch phase) +
      ``complex128`` identity MO so the dtype contract propagates into
      the geminal block.
    """
    spline_coeffs = projection_result.spline_coefficients
    if spline_coeffs.size == 0 or not projection_result.spline_shape:
        raise ValueError(
            "ProjectionResult has no spline block; expected kind='bspline' / "
            "'pgto+bspline' from project_pw_to_spline or project_pw_to_pgto_spline."
        )
    n_band = int(spline_coeffs.shape[0])
    spline_shape = tuple(int(n) for n in projection_result.spline_shape)
    spline_order = int(projection_result.spline_order)
    if k_point_frac is None:
        # QE may choose an arbitrary global phase for every Gamma band.
        # Remove that phase before entering the established real Gamma
        # representation.  Degenerate bands may still be mixed by a complex
        # unitary; as on the pGTO path, the real projection is the Gamma
        # convention and we warn when the remaining imaginary component is
        # appreciable.  The multi-twist writer does not use this branch: it
        # requests the uniform complex k=(0,0,0) representation instead.
        coeffs_complex = np.asarray(spline_coeffs, dtype=np.complex128).reshape(n_band, -1).copy()
        for band in range(n_band):
            row = coeffs_complex[band]
            if row.size == 0:
                continue
            pivot = row[int(np.argmax(np.abs(row)))]
            if abs(pivot) > 0.0:
                row *= np.exp(-1j * np.angle(pivot))
        max_abs_im = float(np.max(np.abs(coeffs_complex.imag))) if coeffs_complex.size else 0.0
        max_abs_re = float(np.max(np.abs(coeffs_complex.real))) if coeffs_complex.size else 0.0
        if max_abs_im > 1.0e-6 * max(1.0, max_abs_re):
            logger.warning(
                "B-spline projection coefficients at Gamma retain a non-negligible "
                "imaginary component after per-band phase fixing (max |Im| = %.3e, "
                "max |Re| = %.3e); taking the real projection. Degenerate Gamma "
                "bands may differ by a unitary rotation within their subspace.",
                max_abs_im,
                max_abs_re,
            )
        aos = AOs_bspline_data.build(
            structure_data=structure_data,
            spline_coefficients=coeffs_complex.real,
            spline_shape=spline_shape,
            spline_order=spline_order,
        )
        mo_coeffs = np.eye(n_band, dtype=np.float64)
        return aos, mo_coeffs
    # Non-Gamma: keep the spline coefficients complex (the projection
    # already returns complex128 at non-Gamma; casting them to real would
    # drop Im and corrupt psi).
    aos = AOs_bspline_complex_data.build(
        structure_data=structure_data,
        k_point=k_point_frac,
        spline_coefficients=spline_coeffs,
        spline_shape=spline_shape,
        spline_order=spline_order,
    )
    mo_coeffs = np.eye(n_band, dtype=np.complex128)
    return aos, mo_coeffs


# ---------------------------------------------------------------------------
# pGTO AO leaf assembly
# ---------------------------------------------------------------------------
def _build_pgto_leaf_and_mo(
    projection_result: ProjectionResult,
    structure_data: Structure_data,
    atom_positions: npt.NDArray[np.float64],
    k_point_frac: tuple[float, float, float] | None,
) -> tuple[AOs_cart_pgto_data, npt.NDArray]:
    """Build an :class:`AOs_cart_pgto_data` leaf + per-band MO coefficients.

    Each :class:`CartPrimitive` becomes one jQMC AO with one primitive
    (uncontracted, ``coefficient = 1``); the projection coefficients are
    rescaled by the jQMC Cartesian-GTO normalization so the per-band
    reconstruction matches.

    Return dtype: ``np.float64`` at Gamma (``k_point_frac is None``),
    ``np.complex128`` off-Gamma (Bloch bands carry a non-trivial phase
    that must propagate into the geminal).
    """
    prims: tuple[CartPrimitive, ...] = projection_result.primitives
    if not prims or projection_result.coefficients.size == 0:
        raise ValueError("ProjectionResult has no pGTO block; expected a kind that emits primitives.")
    num_ao = len(prims)
    nucleus_index: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    angular_momentums: list[int] = []
    polynominal_order_x: list[int] = []
    polynominal_order_y: list[int] = []
    polynominal_order_z: list[int] = []
    normalisations = np.empty(num_ao, dtype=np.float64)
    for mu, prim in enumerate(prims):
        # Find the atom index whose Cartesian position matches the primitive
        # center. pGTO primitives emitted by ``build_pgto_primitives`` are
        # seated on the atomic centers (one primitive per (atom, shell,
        # exponent, component)), so the matching is exact up to
        # floating-point error.
        _center = np.asarray(prim.center, dtype=np.float64)
        atom_idx = None
        for _i, _pos in enumerate(atom_positions):
            if np.allclose(
                _center, np.asarray(_pos, dtype=np.float64), atol=ATOL_PGTO_PRIMITIVE_CENTER_MATCH, rtol=RTOL_BIT_EXACT
            ):
                atom_idx = int(_i)
                break
        if atom_idx is None:
            raise ValueError(f"pGTO primitive center {prim.center} does not match any atomic position.")
        nucleus_index.append(atom_idx)
        exponents.append(float(prim.exponent))
        coefficients.append(1.0)
        nx, ny, nz = prim.ang
        polynominal_order_x.append(int(nx))
        polynominal_order_y.append(int(ny))
        polynominal_order_z.append(int(nz))
        angular_momentums.append(int(nx + ny + nz))
        # Normalization factor used by jQMC Cartesian (p)GTO leaves. The
        # jQMC Cartesian GTO leaf evaluates a primitive as
        # ``N_n_l * exp(-alpha r^2) * x^nx y^ny z^nz`` with
        #
        #     N_n_l = sqrt( (2 alpha / pi)^{3/2} * (8 alpha)^l
        #                   * nx! ny! nz! / ((2 nx)! (2 ny)! (2 nz)!) ),
        #
        # matching ``_compute_AOs_cart_debug``. The pGTO basis used by
        # ``jqmc.pw_pgto_projection`` is unnormalised (a raw
        # ``x^nx y^ny z^nz exp(-alpha r^2)`` primitive), so projection
        # coefficients must be divided by ``N_n_l`` when populating the
        # jQMC MO matrix.
        _l = int(nx) + int(ny) + int(nz)
        _fact = scipy.special.factorial
        _radial = (2.0 * float(prim.exponent) / np.pi) ** 1.5 * (8.0 * float(prim.exponent)) ** _l
        _angular = (
            _fact(int(nx)) * _fact(int(ny)) * _fact(int(nz)) / (_fact(2 * int(nx)) * _fact(2 * int(ny)) * _fact(2 * int(nz)))
        )
        normalisations[mu] = float(np.sqrt(_radial * _angular))

    # Dispatch the AO leaf on ``k_point_frac``: at a non-Gamma twist the
    # leaf must be ``AOs_cart_pgto_complex_data`` (Bloch image-phase factors
    # live on the leaf itself); at Gamma we use the real
    # ``AOs_cart_pgto_data`` builder.
    _aos_kwargs = dict(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=num_ao,
        num_ao_prim=num_ao,
        orbital_indices=tuple(range(num_ao)),
        exponents=np.asarray(exponents, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        angular_momentums=tuple(angular_momentums),
        polynominal_order_x=tuple(polynominal_order_x),
        polynominal_order_y=tuple(polynominal_order_y),
        polynominal_order_z=tuple(polynominal_order_z),
    )
    if k_point_frac is None:
        aos = AOs_cart_pgto_data.build(**_aos_kwargs)
    else:
        aos = build_AOs_cart_pgto_complex_data(**_aos_kwargs, k_point=k_point_frac)
    # Divide projection coeffs by per-AO normalization (jqmc multiplies by
    # N_n_l inside the kernel). At a non-Gamma twist the projection
    # coefficients are genuinely complex (the Bloch envelope carries a
    # per-G phase), so carry complex through to the geminal where the
    # dtype-polymorphic ``MOs_data`` consumes it. Bit-identical at Gamma
    # where ``proj_coeffs`` is real-valued (``Im == 0``).
    proj_coeffs = np.asarray(projection_result.coefficients, dtype=np.complex128)
    mo_coeffs_complex = proj_coeffs / normalisations[None, :]
    is_gamma = k_point_frac is None
    max_abs_im = float(np.max(np.abs(mo_coeffs_complex.imag))) if mo_coeffs_complex.size else 0.0
    if is_gamma:
        # Gamma: Im should vanish to round-off. Anything bigger is a
        # signal that the upstream projection went wrong (or the user
        # forgot to set k_point in structure_data).
        if max_abs_im > 1.0e-6:
            logger.warning(
                "pGTO projection coefficients at Gamma carry non-negligible "
                "imaginary part (max |Im| = %.3e); taking the real part "
                "(Gamma-point bands should be real up to a global phase).",
                max_abs_im,
            )
        mo_coeffs = mo_coeffs_complex.real.astype(np.float64)
    else:
        # Off-Gamma: complex by construction (downstream the dn channel
        # is built as conj(up)). Pass through as complex128.
        mo_coeffs = mo_coeffs_complex.astype(np.complex128)
    return aos, mo_coeffs


# ---------------------------------------------------------------------------
# Composite assembly
# ---------------------------------------------------------------------------
def _build_pgto_pw_composite(
    qe_reference: QEReference,
    projection_result: ProjectionResult,
    structure_data: Structure_data,
    k_point_frac: tuple[float, float, float] | None,
) -> tuple[AOs_composite_data, npt.NDArray]:
    """``pGTO + PW`` mixed leaf; MO row = ``[pgto_coeffs, pw_real_coeffs]``.

    Return dtype: ``np.float64`` at Gamma, ``np.complex128`` off-Gamma.
    The downstream ``MOs_data`` is dtype-polymorphic so the composite
    mo_coeffs flow through unchanged.
    """
    pgto_leaf, pgto_mo = _build_pgto_leaf_and_mo(projection_result, structure_data, qe_reference.atom_positions, k_point_frac)
    pw_leaf, pw_mo = _build_pw_leaf_and_mo(qe_reference, projection_result, structure_data, k_point_frac)
    composite = pgto_leaf + pw_leaf  # AOs_composite_data (flattened by __add__)
    composite.sanity_check()
    # Preserve the widest dtype across the concatenated blocks (complex
    # at non-Gamma); no .astype(float64) hardcast. Bit-identical at Gamma
    # where both inputs are real.
    mo_coeffs = np.concatenate([pgto_mo, pw_mo], axis=1)
    return composite, mo_coeffs


def _build_pgto_bspline_composite(
    projection_result: ProjectionResult,
    structure_data: Structure_data,
    atom_positions: npt.NDArray[np.float64],
    k_point_frac: tuple[float, float, float] | None,
) -> tuple[AOs_composite_data, npt.NDArray]:
    """``pGTO + per-band spline`` mixed leaf.

    The spline AO block here has one AO per band (it holds the *smooth
    residual* ``psi - sum_mu c_mu g_mu``), so the per-band MO row is

        [ pgto_coeffs_for_band_b ,  e_b ]

    with ``e_b`` the standard basis vector (the b-th band picks out the b-th
    spline AO).

    Return dtype mirrors :func:`_build_pgto_leaf_and_mo`: real at Gamma,
    complex off-Gamma.
    """
    if k_point_frac is None:
        # A Gamma band's arbitrary phase multiplies BOTH pieces of the mixed
        # reconstruction.  Fix it once on the spline pivot and apply exactly
        # the same factor to the pGTO coefficients; independently realifying
        # the two blocks would change ``pGTO + residual`` scientifically.
        spline_rows = np.asarray(projection_result.spline_coefficients, dtype=np.complex128).reshape(
            int(projection_result.spline_coefficients.shape[0]), -1
        )
        phase = np.ones(spline_rows.shape[0], dtype=np.complex128)
        for band, row in enumerate(spline_rows):
            if row.size:
                pivot = row[int(np.argmax(np.abs(row)))]
                if abs(pivot) > 0.0:
                    phase[band] = np.exp(-1j * np.angle(pivot))
        projection_result = replace(
            projection_result,
            coefficients=phase[:, None] * np.asarray(projection_result.coefficients),
            spline_coefficients=phase[:, None] * np.asarray(projection_result.spline_coefficients),
        )

    pgto_leaf, pgto_mo = _build_pgto_leaf_and_mo(projection_result, structure_data, atom_positions, k_point_frac)
    bspline_leaf, _ = _build_bspline_leaf_and_mo(projection_result, structure_data, k_point_frac)
    n_band = int(pgto_mo.shape[0])
    if int(bspline_leaf.num_ao) != n_band:
        raise ValueError(
            f"pgto+bspline composite: spline AO count ({bspline_leaf.num_ao}) must "
            f"match pgto band count ({n_band}); the spline block holds the smooth "
            "residual one-per-band."
        )
    composite = pgto_leaf + bspline_leaf
    composite.sanity_check()
    # Identity block must follow the pgto_mo dtype so the concat does not
    # downcast complex pgto_mo back to real at non-Gamma.
    identity = np.eye(n_band, dtype=pgto_mo.dtype)
    mo_coeffs = np.concatenate([pgto_mo, identity], axis=1)
    return composite, mo_coeffs


# ---------------------------------------------------------------------------
# Band-order MO orthonormalization (pure pGTO kind)
# ---------------------------------------------------------------------------
def _mo_gram_matrix(
    mo_coefficients: npt.NDArray,
    overlap: npt.NDArray,
) -> npt.NDArray:
    r"""``M = conj(C) S C^T`` (rows of ``C`` are MOs), Hermitized, fp64 / complex128."""
    c = np.asarray(mo_coefficients)
    s = np.asarray(overlap)
    dtype = np.complex128 if (np.iscomplexobj(c) or np.iscomplexobj(s)) else np.float64
    c = c.astype(dtype, copy=False)
    s = s.astype(dtype, copy=False)
    gram = c.conj() @ s @ c.T
    return 0.5 * (gram + gram.conj().T)


def _normalized_non_orthogonality(gram: npt.NDArray) -> float:
    r"""``max_ij |M_ij / sqrt(M_ii M_jj) - delta_ij|`` (zero for a single MO).

    Dividing by the per-MO norms removes any common or per-MO scale, so this
    reports the genuine projection non-orthogonality only.
    """
    diagonal = np.real(np.diag(gram))
    if diagonal.size < 2:
        return 0.0
    scale = np.sqrt(np.clip(np.outer(diagonal, diagonal), 0.0, None))
    ratio = np.divide(np.abs(gram), scale, out=np.zeros(scale.shape, dtype=np.float64), where=scale > 0.0)
    np.fill_diagonal(ratio, 0.0)
    return float(np.max(ratio))


def _require_nonsingular_occupied_pgto_mos(
    occupied_mo_coefficients: npt.NDArray,
    overlap: npt.NDArray,
    *,
    label: str,
    projection_result: ProjectionResult | None = None,
) -> None:
    r"""Reject a pGTO occupied MO block whose Slater determinant vanishes identically.

    ``occupied_mo_coefficients`` are the occupied rows that enter the
    determinant (one MO per row) and ``overlap`` is the AO overlap ``S`` of
    their leaf. With the occupied Gram ``M = conj(C) S C^T`` the block is
    rejected (``ValueError``, issue #580) when

    * an occupied MO is identically zero (``M_ii <= 0``; e.g. when every
      canonical-orthogonalisation mode of the fit is dropped), or
    * the per-MO-normalized Gram ``D^{-1/2} M D^{-1/2}`` (``D = diag(M)``) has
      ``lambda_min <= PW_PGTO_SVD_RCOND * lambda_max``, i.e. the occupied MOs
      are linearly dependent by the converter's own fp64 linear-dependence
      floor (the cond <= 1e8 criterion the projection applies to ``S``;
      ``1e-8`` is also ``~ sqrt(eps64)``, so the band-order Cholesky residual
      ``~ cond(M) eps64`` stays below ``~ 2e-8`` for every accepted block).
      This is the case when the fit kept fewer independent modes than
      occupied bands.

    Normalizing by ``D`` makes the test independent of any common or per-MO
    scale, which cancels in every QMC ratio; only linear dependence (the
    determinant being identically zero) is rejected. The residual of a
    full-rank fit is not judged here.
    """
    c = np.asarray(occupied_mo_coefficients)
    if c.shape[0] == 0:
        return
    gram = _mo_gram_matrix(c, overlap)
    norms2 = np.real(np.diag(gram))
    if projection_result is not None:
        fit = (
            f" The pGTO fit kept {int(projection_result.n_gto) + int(projection_result.n_pw) - int(projection_result.n_dropped)} "
            f"of {int(projection_result.n_gto) + int(projection_result.n_pw)} overlap modes "
            f"(dropped {int(projection_result.n_dropped)}, cond(S) after truncation {float(projection_result.s_cond):.3e})."
        )
    else:
        fit = ""
    remedy = (
        " The Slater determinant of this block vanishes identically, so no Hamiltonian is written. "
        "Raise --max-cond-number (svd_rcond = 1 / max_cond_number drops overlap modes below "
        "svd_rcond * lambda_max) or lower --alpha-min."
    )
    if not np.all(norms2 > 0.0):
        zero_rows = np.flatnonzero(~(norms2 > 0.0)).tolist()
        raise ValueError(
            f"QE -> jQMC pGTO conversion ({label}): occupied MO rows {zero_rows} of {int(c.shape[0])} are "
            f"identically zero after the projection.{fit}{remedy}"
        )
    scale = 1.0 / np.sqrt(norms2)
    eigenvalues = np.linalg.eigvalsh(scale[:, None] * gram * scale[None, :])
    if not eigenvalues[0] > PW_PGTO_SVD_RCOND * eigenvalues[-1]:
        raise ValueError(
            f"QE -> jQMC pGTO conversion ({label}): the {int(c.shape[0])} occupied MOs are linearly dependent "
            f"(normalized occupied Gram eigenvalues in [{float(eigenvalues[0]):.3e}, {float(eigenvalues[-1]):.3e}], "
            f"ratio below the linear-dependence floor {PW_PGTO_SVD_RCOND:.0e}).{fit}{remedy}"
        )


def _orthonormalize_mo_coefficients_band_order(
    mo_coefficients: npt.NDArray,
    overlap: npt.NDArray,
    *,
    label: str = "MO block",
) -> tuple[npt.NDArray, bool]:
    r"""Band-order (Cholesky) orthonormalization of MO rows in the AO metric ``S``.

    ``mo_coefficients`` is ``(num_mo, num_ao)`` with one MO per row
    (``phi_i = sum_mu C[i, mu] chi_mu``) and ``overlap`` is the AO overlap
    ``S_mu,nu = integral conj(chi_mu) chi_nu`` of the SAME AO leaf (the
    :func:`compute_overlap_matrix` convention). With
    ``M = conj(C) S C^T = L L^H`` (``L`` lower triangular) the result is

        ``C' = solve(conj(L), C) = conj(L)^{-1} C``,

    which gives ``conj(C') S C'^T = I``. ``conj(L)^{-1}`` is lower
    triangular, so new MO ``i`` is a combination of old MOs ``0..i`` only:
    the span of every lowest-N subset is unchanged and a determinant of the
    lowest N rows changes by the constant ``det(conj(L)^{-1})[:N, :N]``. A
    real ``C`` and real ``S`` (Gamma) stay real. Never ``L^{-1} C`` (wrong at
    a general k, invisible at Gamma) and never Loewdin ``M^{-1/2}`` (mixes
    all bands and changes the occupied subspace).

    The normalized non-orthogonality of the input
    (``max_ij |M_ij / sqrt(M_ii M_jj) - delta_ij|``) and the residual
    ``max|conj(C') S C'^T - I|`` are logged at info level.

    Returns:
        ``(coefficients, applied)``: the orthonormalized coefficients (float64
        when ``C`` and ``S`` are both real, complex128 otherwise) and ``True``,
        or -- when ``M`` is not positive definite
        (linearly dependent projected MOs, Cholesky fails) -- the unchanged
        input and ``False`` after a ``logger.warning``.
    """
    c_in = np.asarray(mo_coefficients)
    if c_in.ndim != 2 or c_in.shape[1] != np.asarray(overlap).shape[0]:
        raise ValueError(
            f"_orthonormalize_mo_coefficients_band_order: MO block shape {c_in.shape} does not match "
            f"the AO overlap shape {np.asarray(overlap).shape}."
        )
    if c_in.shape[0] == 0:
        return c_in, False
    gram = _mo_gram_matrix(c_in, overlap)
    diagonal = np.real(np.diag(gram))
    non_orthogonality = _normalized_non_orthogonality(gram)
    try:
        lower = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        logger.warning(
            "MO orthonormalization skipped for %s (%d MOs): the MO Gram matrix conj(C) S C^T is not "
            "positive definite (linearly dependent projected MOs); the block is left unchanged. "
            "Normalized non-orthogonality %.3e, min diagonal %.3e.",
            label,
            int(c_in.shape[0]),
            non_orthogonality,
            float(np.min(diagonal)),
        )
        return c_in, False
    work = c_in.astype(gram.dtype, copy=False)
    c_out = scipy.linalg.solve_triangular(lower.conj(), work, lower=True)
    residual = float(np.max(np.abs(_mo_gram_matrix(c_out, overlap) - np.eye(c_out.shape[0]))))
    logger.info(
        "MO orthonormalization (%s, %d MOs, band-order Cholesky in the AO metric): "
        "normalized non-orthogonality before %.3e, MO norms^2 in [%.6e, %.6e] before, "
        "max|conj(C) S C^T - I| after %.3e.",
        label,
        int(c_in.shape[0]),
        non_orthogonality,
        float(np.min(diagonal)),
        float(np.max(diagonal)),
        residual,
    )
    # float64 when both C and S are real (Gamma), complex128 otherwise.
    return np.asarray(c_out, dtype=gram.dtype), True


def _checked_pgto_block_overlap(
    aos_data: AOs_base_data,
    mo_coeffs: npt.NDArray,
    projection_result: ProjectionResult,
    *,
    label: str,
    overlap: npt.NDArray | None = None,
) -> npt.NDArray:
    """Return ``S`` of one pure-pGTO ``(k, sigma)`` leaf after rejecting a singular occupied block.

    ``S`` is :func:`compute_overlap_matrix` of ``aos_data`` (or the supplied
    ``overlap`` of an identical leaf). The occupied rows of ``mo_coeffs``
    (projection occupation > 0.5, the rows the determinant uses) must pass
    :func:`_require_nonsingular_occupied_pgto_mos`, with or without the
    orthonormalization step that follows.
    """
    if overlap is None:
        overlap = np.asarray(compute_overlap_matrix(aos_data))
    occupied = np.flatnonzero(np.asarray(projection_result.occupations, dtype=np.float64) > 0.5)
    _require_nonsingular_occupied_pgto_mos(
        np.asarray(mo_coeffs)[occupied], overlap, label=label, projection_result=projection_result
    )
    return overlap


def _orthonormalize_pgto_block(
    aos_data: AOs_base_data,
    mo_coeffs: npt.NDArray,
    projection_result: ProjectionResult,
    *,
    label: str,
    overlap: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Orthonormalize one ``(k, sigma)`` pure-pGTO MO block; return ``(mo_coeffs, S)``.

    ``S`` is :func:`compute_overlap_matrix` of the jQMC AO leaf actually built
    for this block (so normalization, ordering and the Bloch phase match the
    AOs jQMC evaluates); an ``overlap`` already computed for an identical leaf
    (the other LSDA spin at the same k) is reused instead. The projected bands
    must be in ascending QE band order, the order in which the Cholesky
    factorization preserves every lowest-N subspace.
    """
    band_indices = np.asarray(projection_result.band_indices, dtype=np.int64)
    if band_indices.size > 1 and not np.all(np.diff(band_indices) > 0):
        raise ValueError(
            f"MO orthonormalization ({label}): projected band indices must be strictly ascending "
            f"(QE energy order), got {band_indices.tolist()}."
        )
    occupied = np.flatnonzero(np.asarray(projection_result.occupations, dtype=np.float64) > 0.5)
    if not np.array_equal(occupied, np.arange(occupied.size)):
        logger.warning(
            "MO orthonormalization (%s): the occupied rows %s are not the lowest bands, so the "
            "band-order Cholesky step may change the occupied subspace selected for the determinant.",
            label,
            occupied.tolist(),
        )
    if overlap is None:
        overlap = np.asarray(compute_overlap_matrix(aos_data))
    mo_out, _ = _orthonormalize_mo_coefficients_band_order(mo_coeffs, overlap, label=label)
    return mo_out, overlap


# ---------------------------------------------------------------------------
# Geminal packaging
# ---------------------------------------------------------------------------
def _make_geminal_from_mos(
    mos_data: MOs_data,
    occupations: npt.NDArray[np.float64],
    n_electrons_total: float,
    *,
    mos_data_dn: MOs_data | None = None,
    occupations_dn: npt.NDArray[np.float64] | None = None,
) -> Geminal_data:
    """Build a canonical occupied-orbital :class:`SD_data` from QE MOs.

    The spin channels always sit at opposite twists ``(+k, -k)``: the up
    channel is the supplied ``+k`` orbitals and the down channel is the
    complex conjugate of its OWN ``+k`` orbitals, i.e. conjugated
    coefficients on the ``with_negated_k()`` AO leaf (both operations
    together: conjugated coefficients on the ``+k`` leaf would build a
    mixed-phase orbital that is neither ``phi^{+k}`` nor ``phi^{-k}``).
    Because the Hamiltonian of each spin channel is real (no vector
    potential, no spin-orbit coupling; a collinear Zeeman field is allowed),
    ``conj(phi^{+k})`` is that channel's ``-k`` orbital.

    Two modes, differing only in coefficient tying:

    - **Restricted (RHF / kRHF / ROHF / kROHF)** -- pass only ``mos_data``.
      The single MO set serves both spins: ``C_dn = conj(C_up)`` on the
      paired core, ``time_reversal=True``. ROHF singly occupied alpha
      orbitals are additional occupied rows of ``mos_data``.
    - **Unrestricted (UHF / kUHF / UKS / kUKS / LSDA)** -- pass both
      ``mos_data`` (alpha) and ``mos_data_dn`` (beta) plus their
      occupations. The channels stay variationally independent
      (``time_reversal=False``); the down channel is ``conj(C_beta)`` on
      the beta basis' ``-k`` leaf, never ``conj(C_alpha)``.

    At Gamma / molecular the conjugation and the ``-k`` leaf are identities.
    """
    is_unrestricted = mos_data_dn is not None
    if is_unrestricted:
        if occupations_dn is None:
            raise ValueError(
                "UKS / UHF path: ``occupations_dn`` must be supplied alongside "
                "``mos_data_dn`` so the loader can derive ``ne_dn`` from the "
                "beta-channel occupations separately."
            )
        # Each spin channel carries its own occupation pattern and thus its
        # own electron count. QMC fixes the electron numbers, so the UKS /
        # UHF path accepts ONLY integer (N_up, N_dn): every band occupation
        # must be binary (0 or 1). The sums below are then exact integers
        # up to round-off.
        check_binary_occupations(occupations, channel="up")
        check_binary_occupations(occupations_dn, channel="dn")
        ne_up = int(round(float(np.asarray(occupations, dtype=np.float64).sum())))
        ne_dn = int(round(float(np.asarray(occupations_dn, dtype=np.float64).sum())))
        if ne_dn > ne_up:
            raise ValueError(
                f"UKS / UHF path: ne_up={ne_up} < ne_dn={ne_dn}. jqmc packs the "
                "excess spin in the up channel (positive magnetization); flip the "
                "spin labels (QE: the sign of tot_magnetization) and reconvert."
            )
        # The source alpha/beta calculations may expose different total band
        # counts. Canonical SD has no square paired-lambda requirement: each
        # spin is selected independently below and only needs enough occupied
        # rows for its own electron count.
    else:
        # Restricted path: same integer-filling contract. The occupied bands
        # feeding the Slater fill must be binary; the electron count itself
        # comes from ``n_electrons_total`` (QE nelec, always integer for a
        # fixed-occupation run).
        check_binary_occupations(occupations)
        # Determine ``(num_electron_up, num_electron_dn)`` for the Slater
        # determinant. The projection returns the occupied bands of a
        # paramagnetic (spin-restricted) QE run; ``n_electrons_total`` is the
        # QE valence electron count. For a closed shell we set
        # ``ne_up = ne_dn = n_electrons_total / 2``; for an odd valence (e.g.
        # Cu with 19 valence electrons) we lift the extra electron into the
        # up channel: ``ne_up = ceil(n / 2)``, ``ne_dn = floor(n / 2)``. The
        # number of "occupied" bands kept by the projection (one MO per
        # band) must be at least ``ne_up`` so the Slater determinant can be
        # filled.
        _n_total_int = int(round(float(n_electrons_total)))
        ne_up = (_n_total_int + 1) // 2
        ne_dn = _n_total_int // 2
    occupied_up = np.flatnonzero(np.asarray(occupations, dtype=np.float64) > 0.5)
    occupied_dn = np.flatnonzero(np.asarray(occupations_dn, dtype=np.float64) > 0.5) if is_unrestricted else occupied_up
    if occupied_up.size < ne_up:
        raise ValueError(
            f"QE -> jQMC: occupation table contains only {occupied_up.size} occupied up bands, "
            f"but the SD requires ne_up={ne_up}."
        )
    if occupied_dn.size < ne_dn:
        raise ValueError(
            f"QE -> jQMC: occupation table contains only {occupied_dn.size} occupied down bands, "
            f"but the SD requires ne_dn={ne_dn}."
        )

    # Occupation, rather than ``nbnd`` or row position, defines the canonical
    # SD rows.  Virtual QE bands never enter SD_data, variational blocks, or
    # the S-Stiefel manifold.
    mos_up_occ = mos_data.replace(
        num_mo=int(ne_up),
        mo_coefficients=np.ascontiguousarray(np.asarray(mos_data.mo_coefficients)[occupied_up[:ne_up]]),
    )
    # Down channel: the complex conjugate of the channel's own +k orbitals,
    # realised as conj(C) on the with_negated_k() leaf. Restricted input ties
    # it to the alpha set; unrestricted input conjugates the independent beta
    # set.
    src_dn = mos_data_dn if is_unrestricted else mos_data
    if hasattr(src_dn.aos_data, "with_negated_k"):
        aos_data_dn = src_dn.aos_data.with_negated_k()
    else:
        aos_data_dn = src_dn.aos_data  # Gamma / molecular
    dn_coefficients = np.asarray(src_dn.mo_coefficients)[occupied_dn[:ne_dn]]
    mos_data_dn_resolved = MOs_data(
        num_mo=int(ne_dn),
        aos_data=aos_data_dn,
        mo_coefficients=np.ascontiguousarray(np.conj(dn_coefficients)),
    )
    _time_reversal_flag = not is_unrestricted
    geminal_data = SD_data.build(
        num_electron_up=int(ne_up),
        num_electron_dn=int(ne_dn),
        mos_up=mos_up_occ,
        mos_dn=mos_data_dn_resolved,
        time_reversal=_time_reversal_flag,
    )
    return geminal_data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _build_leaf_and_mo_for_kind(
    kind: str,
    qe_reference: QEReference,
    projection_result: ProjectionResult,
    structure_data: Structure_data,
    k_point_frac: tuple[float, float, float] | None,
) -> tuple[AOs_base_data, npt.NDArray]:
    """Dispatch one ``(reference, projection)`` pair to its AO leaf + MO block."""
    if kind == "pw":
        return _build_pw_leaf_and_mo(qe_reference, projection_result, structure_data, k_point_frac)
    if kind == "bspline":
        return _build_bspline_leaf_and_mo(projection_result, structure_data, k_point_frac)
    if kind == "pgto":
        return _build_pgto_leaf_and_mo(projection_result, structure_data, qe_reference.atom_positions, k_point_frac)
    if kind == "pgto+pw":
        return _build_pgto_pw_composite(qe_reference, projection_result, structure_data, k_point_frac)
    if kind == "pgto+bspline":
        return _build_pgto_bspline_composite(projection_result, structure_data, qe_reference.atom_positions, k_point_frac)
    raise ValueError(f"Unsupported kind={kind!r}.")


def build_wavefunction_from_projection(
    qe_reference: QEReference,
    projection_result: ProjectionResult,
    *,
    kind: str,
    qe_reference_dn: QEReference | None = None,
    projection_result_dn: ProjectionResult | None = None,
    force_complex_gamma: bool = False,
    orthonormalize: bool = True,
) -> tuple[Structure_data, AOs_base_data, Geminal_data]:
    """Assemble jqmc data from a QE projection result.

    Args:
        qe_reference: Loaded QE reference (cell, atoms, bands). For an LSDA
            (``nspin = 2``) run this is the **up-spin** channel.
        projection_result: Output of one of the ``project_pw_to_*`` functions
            in :mod:`jqmc.pw_pgto_projection` for ``qe_reference``.
        kind: One of ``{"pw", "bspline", "pgto", "pgto+pw", "pgto+bspline"}``,
            matching the projection routine that produced ``projection_result``.
        qe_reference_dn: Optional **down-spin** channel QE reference of an
            LSDA (``nspin = 2``) run; must be given together with
            ``projection_result_dn``. Enables the unrestricted (UKS / UHF)
            geminal: each spin channel gets its own AO leaf + MO block,
            ``time_reversal = False``, and ``ne_up`` / ``ne_dn`` are derived
            from the per-channel occupations. The source channels need not
            carry the same total band count; each only needs enough occupied
            bands for its own electron count.
        projection_result_dn: Projection result for ``qe_reference_dn``
            (the down-spin channel); must be given together with
            ``qe_reference_dn``.
        force_complex_gamma: Encode Gamma as the complex Bloch leaf with
            ``k_point=(0,0,0)``. This is reserved for the multi-twist combined
            writer, which requires one uniform representation across Gamma
            and non-Gamma entries. The default ``False`` preserves the
            established single-k real Gamma path.
        orthonormalize: For ``kind="pgto"`` only: orthonormalize
            each ``(k, sigma)`` MO block in the true periodic AO metric
            ``S = compute_overlap_matrix(<built pGTO leaf>)`` by band-order
            Cholesky (``C' = conj(L)^{-1} C`` with ``conj(C) S C^T = L L^H``),
            after the per-AO normalization and the Gamma real-part step and
            before ``MOs_data`` is built. The span of every lowest-N band
            subset (hence the determinant up to a constant) is unchanged. The
            LSDA dn block reuses the up-spin ``S`` of the same k; the
            restricted dn channel is the conjugate of the orthonormalized up
            block and is not orthonormalized again. Ignored for ``"pw"``,
            ``"bspline"``, ``"pgto+pw"`` and ``"pgto+bspline"`` (never
            orthonormalized). Default ``True``.

    MO scale: every kind is unit-cell normalized; see the module
    docstring.

    Returns:
        Tuple ``(structure_data, aos_data, geminal_data)``:
            * ``structure_data``: periodic :class:`Structure_data` for the cell.
            * ``aos_data``: AO leaf (or composite) for the chosen ``kind``
              (the up-spin leaf on the LSDA path; the dn leaf lives on
              ``geminal_data.orb_data_dn_spin.aos_data``).
            * ``geminal_data``: Slater-via-Geminal wrapper that places one
              occupied MO per electron pair.
    """
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"kind={kind!r} not in {_SUPPORTED_KINDS}.")
    if (qe_reference_dn is None) != (projection_result_dn is None):
        raise ValueError("LSDA path: qe_reference_dn and projection_result_dn must be supplied together.")

    structure_data, k_point_frac = _build_structure_data(qe_reference)
    if force_complex_gamma and k_point_frac is None:
        # combined twist-average files intentionally use one representation for all
        # twists.  A zero tuple is distinct from ``None``: it selects complex
        # Bloch leaves while retaining a unit Bloch phase at Gamma.
        k_point_frac = (0.0, 0.0, 0.0)

    aos_data, mo_coeffs = _build_leaf_and_mo_for_kind(kind, qe_reference, projection_result, structure_data, k_point_frac)

    aos_data.sanity_check()
    # Band-order Cholesky orthonormalization of the pure pGTO block.
    # Mixed kinds are deliberately excluded (orthonormalizing only their pGTO
    # part would break the pGTO / PW (spline) coefficient ratio).
    _orthonormalize_block = bool(orthonormalize) and kind == "pgto"
    overlap_up = None
    _label_up = "pgto up" if projection_result_dn is not None else "pgto"
    if kind == "pgto":
        # Issue #580: a fit that kept fewer independent modes than occupied
        # bands (or none) leaves a zero / rank-deficient occupied block whose
        # determinant vanishes identically; reject it with or without the
        # orthonormalization step.
        overlap_up = _checked_pgto_block_overlap(aos_data, mo_coeffs, projection_result, label=_label_up)
    if _orthonormalize_block:
        mo_coeffs, overlap_up = _orthonormalize_pgto_block(
            aos_data,
            mo_coeffs,
            projection_result,
            label=_label_up,
            overlap=overlap_up,
        )
    n_band = int(mo_coeffs.shape[0])
    # Preserve the MO coefficient dtype --
    # ``_build_pgto_leaf_and_mo`` (and the pgto composites) already return
    # ``complex128`` off-Gamma; a ``dtype=np.float64`` hardcast here would
    # silently drop ``Im``. ``MOs_data`` is dtype-polymorphic via
    # ``_mo_coefficients_jnp`` so the complex storage flows downstream
    # unchanged. Bit-identical at Gamma where ``mo_coeffs`` is real.
    mos_data = MOs_data(
        num_mo=n_band,
        aos_data=aos_data,
        mo_coefficients=np.asarray(mo_coeffs),
    )
    mos_data.sanity_check()

    if projection_result_dn is not None:
        # LSDA / UKS: independent dn-channel leaf + MO block. The dn wfc
        # shares the cell / atoms / k-point with the up channel (same SCF),
        # so the structure_data and k_point_frac derived above apply.
        aos_data_dn, mo_coeffs_dn = _build_leaf_and_mo_for_kind(
            kind, qe_reference_dn, projection_result_dn, structure_data, k_point_frac
        )
        aos_data_dn.sanity_check()
        if kind == "pgto":
            # S_AO(k) is spin independent: the dn leaf is built from the same
            # primitives at the same k, so the up-spin overlap is reused. Each
            # spin block is checked (issue #580) and orthonormalized
            # independently.
            same_leaf = tuple(projection_result_dn.primitives) == tuple(projection_result.primitives)
            overlap_dn = _checked_pgto_block_overlap(
                aos_data_dn,
                mo_coeffs_dn,
                projection_result_dn,
                label="pgto dn",
                overlap=overlap_up if same_leaf else None,
            )
        if _orthonormalize_block:
            mo_coeffs_dn, _ = _orthonormalize_pgto_block(
                aos_data_dn,
                mo_coeffs_dn,
                projection_result_dn,
                label="pgto dn",
                overlap=overlap_dn,
            )
        n_band_dn = int(mo_coeffs_dn.shape[0])
        mos_data_dn = MOs_data(
            num_mo=n_band_dn,
            aos_data=aos_data_dn,
            mo_coefficients=np.asarray(mo_coeffs_dn),
        )
        mos_data_dn.sanity_check()
        geminal_data = _make_geminal_from_mos(
            mos_data,
            np.asarray(projection_result.occupations, dtype=np.float64),
            float(qe_reference.bands.n_electrons),
            mos_data_dn=mos_data_dn,
            occupations_dn=np.asarray(projection_result_dn.occupations, dtype=np.float64),
        )
    else:
        geminal_data = _make_geminal_from_mos(
            mos_data,
            np.asarray(projection_result.occupations, dtype=np.float64),
            float(qe_reference.bands.n_electrons),
        )

    logger.info(
        "build_wavefunction_from_projection(kind=%s%s): %d AOs, %d MOs, ne_up=%d, ne_dn=%d.",
        kind,
        "" if projection_result_dn is None else ", LSDA",
        int(aos_data.num_ao),
        n_band,
        geminal_data.num_electron_up,
        geminal_data.num_electron_dn,
    )
    return structure_data, aos_data, geminal_data


def build_tiled_bspline_wavefunction(
    qe_references: Sequence[QEReference],
    projection_results: Sequence[ProjectionResult],
    *,
    tiling_matrix: npt.ArrayLike,
    qe_references_dn: Sequence[QEReference] | None = None,
    projection_results_dn: Sequence[ProjectionResult] | None = None,
) -> tuple[Structure_data, AOs_bspline_tiled_complex_data, SD_data]:
    r"""Unfold primitive multi-k B-splines into one Gamma supercell SD.

    The input must contain exactly ``|det(S)|`` primitive k points satisfying
    ``S @ k = 0 (mod Z^3)``.  Only occupied spline bands enter the returned
    canonical :class:`SD_data`; the spline grid remains the primitive grid.
    Restricted inputs currently require an even primitive electron count so
    every k block has the same closed-shell occupied rows.  LSDA inputs carry
    independent up/down tiled leaves and may have unequal spin populations.
    """
    matrix = normalize_tiling_matrix(tiling_matrix)
    determinant = abs(int(round(np.linalg.det(matrix.astype(np.float64)))))
    references = tuple(qe_references)
    projections = tuple(projection_results)
    if len(references) != determinant or len(projections) != determinant:
        raise ValueError(
            f"tiled B-spline conversion requires exactly |det(S)|={determinant} primitive k blocks; "
            f"got {len(references)} references and {len(projections)} projections."
        )
    unrestricted = qe_references_dn is not None or projection_results_dn is not None
    if (qe_references_dn is None) != (projection_results_dn is None):
        raise ValueError("tiled LSDA conversion requires both qe_references_dn and projection_results_dn.")
    references_dn = tuple(qe_references_dn or ())
    projections_dn = tuple(projection_results_dn or ())
    if unrestricted and (len(references_dn) != determinant or len(projections_dn) != determinant):
        raise ValueError(f"tiled LSDA conversion requires {determinant} down-spin k blocks.")

    primitive_structure, _ = _build_structure_data(references[0])
    supercell_structure = tile_structure_data(primitive_structure, matrix)
    k_points: list[tuple[float, float, float]] = []
    up_blocks: list[np.ndarray] = []
    down_blocks: list[np.ndarray] = []
    spline_shape: tuple[int, int, int] | None = None
    spline_order: int | None = None
    expected_up_per_k: int | None = None
    expected_dn_per_k: int | None = None

    def _occupied_block(
        reference: QEReference,
        projection: ProjectionResult,
        *,
        channel: str,
    ) -> tuple[np.ndarray, int]:
        if projection.spline_coefficients.size == 0 or not projection.spline_shape:
            raise ValueError(f"tiled {channel} projection has no B-spline coefficient block.")
        occupations = np.asarray(projection.occupations, dtype=np.float64)
        check_binary_occupations(occupations, channel=channel)
        occupied = np.flatnonzero(occupations > 0.5)
        if occupied.size == 0:
            raise ValueError(f"tiled {channel} projection has no occupied bands.")
        coefficients = np.asarray(projection.spline_coefficients, dtype=np.complex128)
        return np.ascontiguousarray(coefficients[occupied]), int(occupied.size)

    for index, (reference, projection) in enumerate(zip(references, projections, strict=True)):
        structure_k, k_point = _build_structure_data(reference)
        if not (
            np.allclose(
                structure_k.cell,
                primitive_structure.cell,
                rtol=RTOL_BIT_EXACT,
                atol=ATOL_PYTHON_FP64_LOOSE,
            )
            and np.allclose(
                structure_k.positions,
                primitive_structure.positions,
                rtol=RTOL_BIT_EXACT,
                atol=ATOL_PYTHON_FP64_LOOSE,
            )
            and structure_k.atomic_numbers == primitive_structure.atomic_numbers
        ):
            raise ValueError(f"primitive k block {index} describes a different structure.")
        k = (0.0, 0.0, 0.0) if k_point is None else tuple(float(value) for value in k_point)
        mismatch = matrix @ np.asarray(k, dtype=np.float64)
        mismatch -= np.rint(mismatch)
        if float(np.max(np.abs(mismatch))) > ATOL_TILING_INTEGER_RESIDUE:
            raise ValueError(
                f"primitive k point {k} does not unfold to Gamma for tiling matrix {matrix.tolist()}; "
                "require S @ k to be an integer reciprocal vector."
            )
        k_points.append(k)
        block, occupied_count = _occupied_block(reference, projection, channel=f"up k[{index}]")
        up_blocks.append(block)
        if expected_up_per_k is None:
            expected_up_per_k = occupied_count
        elif occupied_count != expected_up_per_k:
            raise ValueError("tiled conversion requires the same number of occupied up bands at every primitive k point.")
        shape_k = tuple(int(value) for value in projection.spline_shape)
        order_k = int(projection.spline_order)
        if spline_shape is None:
            spline_shape, spline_order = shape_k, order_k
        elif shape_k != spline_shape or order_k != spline_order:
            raise ValueError("all primitive k projections must use the same B-spline shape and order.")

        if unrestricted:
            reference_dn = references_dn[index]
            projection_dn = projections_dn[index]
            _, k_point_dn = _build_structure_data(reference_dn)
            k_dn = (0.0, 0.0, 0.0) if k_point_dn is None else tuple(float(value) for value in k_point_dn)
            if not np.allclose(
                np.asarray(k_dn) - np.asarray(k),
                np.rint(np.asarray(k_dn) - np.asarray(k)),
                rtol=RTOL_BIT_EXACT,
                atol=ATOL_TILING_INTEGER_RESIDUE,
            ):
                raise ValueError(f"LSDA up/down k points differ at block {index}: {k} vs {k_dn}.")
            block_dn, occupied_dn = _occupied_block(reference_dn, projection_dn, channel=f"down k[{index}]")
            if (
                tuple(int(value) for value in projection_dn.spline_shape) != spline_shape
                or int(projection_dn.spline_order) != spline_order
            ):
                raise ValueError("LSDA up/down primitive k projections must share B-spline shape and order.")
            down_blocks.append(block_dn)
            if expected_dn_per_k is None:
                expected_dn_per_k = occupied_dn
            elif occupied_dn != expected_dn_per_k:
                raise ValueError("tiled conversion requires the same number of occupied down bands at every primitive k point.")

    # The leaf also validates distinct coset representatives and the exact
    # S*k congruence.  Sorting is deliberately not applied: QE band/k order is
    # preserved in the determinant rows.
    assert spline_shape is not None and spline_order is not None
    tiled_up = AOs_bspline_tiled_complex_data.build(
        structure_data=supercell_structure,
        spline_coefficients_by_k=up_blocks,
        spline_shape=spline_shape,
        tiling_matrix=matrix,
        k_points=k_points,
        spline_order=spline_order,
        supercell_k_point=(0.0, 0.0, 0.0),
    )
    mos_up = MOs_data.build_identity(aos_data=tiled_up)

    if unrestricted:
        tiled_dn = AOs_bspline_tiled_complex_data.build(
            structure_data=supercell_structure,
            spline_coefficients_by_k=down_blocks,
            spline_shape=spline_shape,
            tiling_matrix=matrix,
            k_points=k_points,
            spline_order=spline_order,
            supercell_k_point=(0.0, 0.0, 0.0),
        ).with_negated_k()
        # Opposite spin twists: the down channel is the complex conjugate of
        # its own beta bands (negated primitive k, conjugated control points).
        mos_dn = MOs_data.build_identity(aos_data=tiled_dn)
        geminal = SD_data.build(
            num_electron_up=tiled_up.num_ao,
            num_electron_dn=tiled_dn.num_ao,
            mos_up=mos_up,
            mos_dn=mos_dn,
            time_reversal=False,
        )
    else:
        primitive_electrons = int(round(float(references[0].bands.n_electrons)))
        if primitive_electrons % 2:
            raise ValueError(
                "restricted tiled B-spline conversion currently requires an even primitive electron count; "
                "use an LSDA QE calculation for an open-shell supercell."
            )
        expected_spin = determinant * (primitive_electrons // 2)
        if tiled_up.num_ao != expected_spin:
            raise ValueError(
                f"tiled occupied-band count {tiled_up.num_ao} does not match the closed-shell supercell "
                f"electron count per spin {expected_spin}."
            )
        geminal = SD_data.build(
            num_electron_up=expected_spin,
            num_electron_dn=expected_spin,
            mos_up=mos_up,
            time_reversal=True,
        )

    logger.info(
        "Built tiled B-spline Gamma supercell: S=%s, |det S|=%d, primitive grid=%s, "
        "unfolded bands=(%d up, %d down), stored spline rows=%d.",
        matrix.tolist(),
        determinant,
        spline_shape,
        geminal.num_electron_up,
        geminal.num_electron_dn,
        tiled_up.num_ao + (0 if not unrestricted else tiled_dn.num_ao),
    )
    return supercell_structure, tiled_up, geminal


# ---------------------------------------------------------------------------
# Hamiltonian assembly (QE path)
# ---------------------------------------------------------------------------
def _build_jastrow_data_from_structure(
    structure_data: Structure_data,
    *,
    j1_param: float | None,
    j1_type: str,
    j2_param: float | None,
    j2_type: str,
    core_electrons: tuple[float, ...] | None = None,
) -> Jastrow_data:
    """Assemble a minimal :class:`Jastrow_data` (one- and/or two-body only).

    The QE path does not carry ECP information (the ``QEReference`` only
    exposes the band structure / valence count), so the one-body Jastrow
    ``core_electrons`` defaults to zero per atom when ``core_electrons`` is
    ``None``. The three-body term is not constructed here; callers that
    need J3 can attach it to the returned :class:`Jastrow_data` via
    :py:meth:`Jastrow_data.replace`.
    """
    j1 = None
    if j1_param is not None:
        natom = int(structure_data.natom)
        if core_electrons is None:
            core_electrons = tuple([0.0] * natom)
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=float(j1_param),
            structure_data=structure_data,
            core_electrons=core_electrons,
            jastrow_1b_type=j1_type,
        )
    j2 = None
    if j2_param is not None:
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=float(j2_param),
            jastrow_2b_type=j2_type,
            structure_data=structure_data,
        )
    return Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2)


def _zero_ecp_placeholder_block(
    structure_data: Structure_data,
    z_cores: tuple[float, ...],
) -> Coulomb_potential_data:
    """Build a zero-coefficient ECP block covering both local and non-local parts.

    The Coulomb_potential_data._{local,non_local}_part_index helpers compute
    ``np.array([...])`` from a Python list comprehension. With an empty list
    the array's dtype defaults to float64, which then breaks the
    integer-indexing downstream (see _padded_parameters_tuple). To preserve
    the int-dtype contract on a system that physically has only a bare
    ECP core charge (e.g. the QE Cu sample), we emit two zero-coefficient
    terms per atom: one local and one non-local (``ang_mom = 0``). Both
    contribute identically zero to every observable.

    l_max = 0 here (a single, contentless non-local channel), so per the
    local-first TREXIO convention the local term is tagged ``l_max + 1 = 1``,
    not 2 -- ``l_max`` counts the non-local channels, and the local one is not
    an l value of its own. Coulomb_potential_data.sanity_check enforces
    that every declared l actually carries a term, so getting this wrong
    here fails loudly instead of silently.
    """
    natom = int(structure_data.natom)
    return Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=tuple(float(z) for z in z_cores),
        max_ang_mom_plus_1=tuple([1] * natom),
        num_ecps=2 * natom,
        ang_moms=tuple([1] * natom + [0] * natom),
        nucleus_index=tuple(list(range(natom)) + list(range(natom))),
        exponents=tuple([1.0] * (2 * natom)),
        coefficients=tuple([0.0] * (2 * natom)),
        powers=tuple([2] * (2 * natom)),
    )


def _build_ecp_coulomb_from_ecp_data(
    structure_data: Structure_data,
    ecp_by_atom: list[ECPData | None],
) -> Coulomb_potential_data:
    """Assemble ECP data for a possibly mixed AE/ECP structure.

    ``ecp_by_atom[i]`` is the parsed analytic ccECP (see
    :func:`jqmc.upf_wrapper.parse_gaussinfo`) for atom ``i`` of
    ``structure_data``. Each atom's flattened
    ``(ang_mom, exponent, coefficient, power)`` terms are concatenated with a
    ``nucleus_index`` tag, exactly mirroring the TREXIO ECP layout consumed
    elsewhere. The local channel is already encoded as
    ``ang_mom = max_ang_mom_plus_1`` per :func:`parse_gaussinfo`.

    Args:
        structure_data: Periodic structure (one ECP per atom).
        ecp_by_atom: Per-atom parsed ccECP, or ``None`` for an all-electron
            atom; ``len == natom``.

    Returns:
        Coulomb_potential_data: ``ecp_flag=True`` block (no Ewald yet).
    """
    natom = int(structure_data.natom)
    if len(ecp_by_atom) != natom:
        raise ValueError(f"ecp_by_atom length {len(ecp_by_atom)} must equal natom={natom}.")

    z_cores: list[float] = []
    max_ang_mom_plus_1: list[int] = []
    ang_moms: list[int] = []
    nucleus_index: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    powers: list[int] = []

    for atom_i, ecp in enumerate(ecp_by_atom):
        if ecp is None:
            z_cores.append(0.0)
            max_ang_mom_plus_1.append(0)
            continue
        z_cores.append(float(ecp.z_core))
        max_ang_mom_plus_1.append(int(ecp.max_ang_mom_plus_1))
        ang_moms.extend(int(a) for a in ecp.ang_moms)
        nucleus_index.extend([atom_i] * ecp.num_ecps)
        exponents.extend(float(e) for e in ecp.exponents)
        coefficients.extend(float(c) for c in ecp.coefficients)
        powers.extend(int(p) for p in ecp.powers)

    return Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=tuple(z_cores),
        max_ang_mom_plus_1=tuple(max_ang_mom_plus_1),
        num_ecps=len(ang_moms),
        ang_moms=tuple(ang_moms),
        nucleus_index=tuple(nucleus_index),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        powers=tuple(powers),
    )


def _build_qe_coulomb_potential_data(
    structure_data: Structure_data,
    n_electron: int,
    ecp_z_cores: tuple[float, ...] | None,
    ecp_by_element: dict[str, ECPData] | None,
) -> tuple[Coulomb_potential_data, int | None]:
    """Build the authoritative QE Coulomb/ECP block before Ewald tables.

    Species present in ``ecp_by_element`` are pseudized; absent species are
    all-electron. This supports mixed AE/ECP structures and zero-core ECPs.
    When analytic ECP records are unavailable, :func:`_build_coulomb_potential_data`
    accepts only explicit core charges or an all-electron (neutral) save and
    raises for a pseudopotential save with no ECP source.
    """
    if ecp_by_element:
        ecp_by_species = {species.lower(): ecp for species, ecp in ecp_by_element.items()}
        ecp_by_atom = [ecp_by_species.get(str(species).lower()) for species in structure_data.atomic_labels]
        if any(ecp is not None for ecp in ecp_by_atom):
            return _build_ecp_coulomb_from_ecp_data(structure_data, ecp_by_atom), int(n_electron)
    return _build_coulomb_potential_data(structure_data, n_electron, ecp_z_cores)


def _upf_species_key(strategy: UPFExponentInit, atom_label: str, atom_element: str) -> str:
    """Return the UPF map key of one QE species: its label (``Fe1``) if registered, else its element.

    This is the label-or-element mapping that :func:`_require_upf_for_every_species`
    validates, so an explicit per-label entry overrides the element entry.
    """
    try:
        strategy.z_valence_for_element(atom_label)
    except KeyError:
        return atom_element
    return atom_label


def _require_upf_for_every_species(
    upf_map: dict[str, str],
    atom_labels: list[str] | tuple[str, ...],
    atom_elements: list[str] | tuple[str, ...],
) -> None:
    """Reject a UPF map that misses a species of the QE structure.

    Every species needs its UPF: it is the source of the analytic ECP (and of
    the all-electron / ECP classification) on every conversion kind, and of
    the pGTO seed on the pGTO-bearing kinds. A species is covered when either
    its QE label (``Fe1``) or its element (``Fe``) is a key, matched
    case-insensitively like :class:`jqmc.upf_wrapper.UPFExponentInit`.

    Raises:
        ValueError: naming the uncovered species and the keys supplied.
    """
    keys = {str(k).strip().capitalize() for k in upf_map}
    missing: list[str] = []
    for label, element in zip(atom_labels, atom_elements, strict=True):
        if str(label).capitalize() in keys or str(element).capitalize() in keys:
            continue
        name = str(label) if str(label) == str(element) else f"{label} ({element})"
        if name not in missing:
            missing.append(name)
    if missing:
        raise ValueError(
            f"No UPF supplied for species {', '.join(missing)} (UPF map keys: {sorted(upf_map)}). "
            "Give one 'element=path/to/element.upf' entry per species (jqmc-tool qe --upf / "
            "QE_Workflow upf_paths), or omit them all to auto-discover the UPFs QE recorded in "
            "<prefix>.save/data-file-schema.xml."
        )


def _build_coulomb_potential_data(
    structure_data: Structure_data,
    n_electron: int,
    ecp_z_cores: tuple[float, ...] | None,
) -> tuple[Coulomb_potential_data, int | None]:
    """Build the Coulomb block when no analytic ECP record is available.

    Two accepted cases:
      * Explicit ``ecp_z_cores`` (one per atom): a core-charge-only block
        (zero-coefficient ECP terms, see :func:`_zero_ecp_placeholder_block`)
        with strict Ewald neutrality. The public conversion routes
        (``run_qe2jqmc`` and :func:`convert_qe_to_twist_average_input`)
        reject positive core charges without a UPF/ECP source before reaching
        here, so this block is only built on explicit request.
      * ``ecp_z_cores is None`` and ``sum(Z) == n_electron``: all-electron;
        Ewald neutrality is enforced.

    ``ecp_z_cores is None`` with ``sum(Z) != n_electron`` raises: the save
    has core electrons removed by a pseudopotential, and without its UPF
    neither the ECP itself nor the per-species core charges are known.
    (Averaging ``(sum(Z) - n_electron) / natom`` into every atom with a
    zero-coefficient block would assume one species, silently drop the ECP
    and mis-assign multi-species charges.)

    Returns:
        ``(coulomb_potential_data, n_electron_for_ewald)`` where the
        second element is the integer to pass to
        :py:func:`build_ewald_data`.

    Raises:
        ValueError: ``ecp_z_cores`` has the wrong length, or core electrons
            are missing and no ECP source was given.
    """
    natom = int(structure_data.natom)
    z_total = float(sum(structure_data.atomic_numbers))

    if ecp_z_cores is not None:
        if len(ecp_z_cores) != natom:
            raise ValueError(f"ecp_z_cores length {len(ecp_z_cores)} must equal natom={natom}.")
        z_cores = tuple(float(z) for z in ecp_z_cores)
        coulomb_potential_data = _zero_ecp_placeholder_block(structure_data, z_cores)
        return coulomb_potential_data, int(n_electron)

    if abs(z_total - float(n_electron)) <= 1.0e-9:
        coulomb_potential_data = Coulomb_potential_data(
            structure_data=structure_data,
            ecp_flag=False,
        )
        return coulomb_potential_data, int(n_electron)

    species = sorted({str(e) for e in structure_data.element_symbols})
    raise ValueError(
        f"QE path: the atoms carry sum(Z)={z_total:g} protons but the save has n_electron={int(n_electron)}: "
        f"core electrons were removed by a pseudopotential, yet no analytic ECP was resolved for "
        f"{', '.join(species)}. Supply each pseudized species' UPF with its [Gaussinfo] ccECP block "
        "(jqmc-tool qe --upf 'element=path/to/element.upf', QE_Workflow upf_paths, or ecp_by_element); "
        "auto-discovery reads the UPFs recorded in <prefix>.save/data-file-schema.xml. "
        "Charged cells (sum(Z) != n_electron without pseudopotentials) are not supported."
    )


def build_hamiltonian_from_wavefunction(
    structure_data: Structure_data,
    aos_data: AOs_base_data,
    geminal_data: Geminal_data,
    *,
    j1_param: float | None = None,
    j1_type: str = "exp",
    j2_param: float | None = 1.0,
    j2_type: str = "pade",
    ewald_tol: float = 1.0e-10,
    ecp_z_cores: tuple[float, ...] | None = None,
    ecp_by_element: dict[str, ECPData] | None = None,
    jastrow_three_body_data: Jastrow_three_body_data | None = None,
    jastrow_dtn_data: Jastrow_DTN_data | None = None,
) -> Hamiltonian_data:
    """Wrap a Structure/AO/Geminal triple into a complete :class:`Hamiltonian_data`.

    Builds:
      * a :class:`Coulomb_potential_data` (analytic ECP for the species in
        ``ecp_by_element``, a core-charge-only block from ``ecp_z_cores``,
        or all-electron bare Coulomb) whose ``ewald_data`` is precomputed
        via :py:func:`build_ewald_data` with ``tol=ewald_tol``, so the
        strict ``pbc_flag iff ewald_data is not None`` contract holds.
      * a :class:`Jastrow_data` carrying optional one-body (``j1_param``),
        two-body (``j2_param``), three-body (``jastrow_three_body_data``)
        and DTN (``jastrow_dtn_data``) terms; the default is a Pade J2
        with ``a = 1.0`` (and no J1, no J3, no DTN).
      * a :class:`Wavefunction_data` combining the geminal and jastrow
        blocks, and the outer :class:`Hamiltonian_data`.

    The AO leaf is referenced unchanged (it is already periodic /
    image-expanded when the QE path delivered it). ``aos_data`` is
    silenced from ``MOs_data.aos_data`` via the geminal's existing MO
    container --- callers should pass the same ``geminal_data`` that
    :func:`build_wavefunction_from_projection` produced.

    Args:
        structure_data: Periodic :class:`Structure_data` for the cell.
        aos_data: AO leaf used by the geminal (kept for API parity with
            :func:`build_wavefunction_from_projection`; not consumed here).
        geminal_data: Slater-via-Geminal wrapper.
        j1_param: Optional one-body Jastrow parameter; ``None`` disables J1.
        j1_type: One-body functional form (``"exp"`` or ``"pade"``).
        j2_param: Optional two-body Jastrow parameter; ``None`` disables J2.
            Defaults to a Pade ``a = 1.0`` so the returned Hamiltonian is
            immediately usable by the QMC drivers.
        j2_type: Two-body functional form (``"pade"`` or ``"exp"``).
        ewald_tol: Convergence tolerance passed to :py:func:`build_ewald_data`.
        ecp_z_cores: Optional per-atom core charges (one per atom), used
            only when ``ecp_by_element`` supplies no ECP for any species;
            builds a core-charge-only Coulomb block. ``None`` requires an
            all-electron (neutral) cell in that case.
        ecp_by_element: Optional analytic ECP records keyed by species
            (case-insensitive). Species present are pseudized, absent ones
            are all-electron; takes precedence over ``ecp_z_cores``.
        jastrow_three_body_data: Optional three-body Jastrow block attached
            to the Jastrow. ``None`` (default) disables J3.
        jastrow_dtn_data: Optional DTN polynomial Jastrow block. Mutually
            exclusive with ``j1_param`` / ``j2_param`` /
            ``jastrow_three_body_data`` (``ValueError`` otherwise).
            ``None`` (default) disables DTN.

    Returns:
        Hamiltonian_data: Periodic, Ewald-ready, with the requested Jastrow
        terms.
    """
    del aos_data  # only used by callers via geminal_data; kept for API parity.
    if not structure_data.pbc_flag:
        raise ValueError(
            "build_hamiltonian_from_wavefunction expects a periodic "
            "structure_data (pbc_flag=True); QE-derived data is always periodic."
        )

    n_electron = int(geminal_data.num_electron_up) + int(geminal_data.num_electron_dn)
    coulomb_potential_data, n_electron_for_ewald = _build_qe_coulomb_potential_data(
        structure_data,
        n_electron,
        ecp_z_cores,
        ecp_by_element,
    )
    core_electrons_for_j1 = tuple(float(z) for z in coulomb_potential_data.z_cores) if coulomb_potential_data.ecp_flag else None

    jastrow_data = _build_jastrow_data_from_structure(
        structure_data,
        j1_param=j1_param,
        j1_type=j1_type,
        j2_param=j2_param,
        j2_type=j2_type,
        core_electrons=core_electrons_for_j1,
    )
    if jastrow_dtn_data is not None and (
        jastrow_data.jastrow_one_body_data is not None
        or jastrow_data.jastrow_two_body_data is not None
        or jastrow_three_body_data is not None
    ):
        # Family-exclusion guard: the DTN poly term carries J1+J2+J3
        # channels itself; mixing it with the Pade J1/J2 + analytic AO J3
        # family is not supported because the Pade ``sin``-distance form
        # cannot be made cusp-exact on oblique cells.
        raise ValueError(
            "jastrow_dtn_data is mutually exclusive with the Pade J1 / J2 "
            "+ analytic AO J3 family. Disable j1_param / j2_param / "
            "jastrow_three_body_data, or pass jastrow_dtn_data=None."
        )
    jastrow_data = jastrow_data.replace(
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_dtn_data=jastrow_dtn_data,
    )
    wavefunction_data = Wavefunction_data(
        jastrow_data=jastrow_data,
        geminal_data=geminal_data,
    )
    ewald_data = build_ewald_data(
        structure_data,
        coulomb_potential_data,
        tol=ewald_tol,
        n_electron=n_electron_for_ewald,
    )
    coulomb_potential_data = coulomb_potential_data.replace(ewald_data=ewald_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()
    logger.info(
        "build_hamiltonian_from_wavefunction: pbc=%s, ne_up=%d, ne_dn=%d, j1=%s, j2=%s.",
        structure_data.pbc_flag,
        geminal_data.num_electron_up,
        geminal_data.num_electron_dn,
        "off" if j1_param is None else f"{j1_type}@{j1_param}",
        "off" if j2_param is None else f"{j2_type}@{j2_param}",
    )
    if jastrow_three_body_data is not None:
        logger.info("build_hamiltonian_from_wavefunction: analytic J3 attached.")
    if jastrow_dtn_data is not None:
        logger.info(
            "build_hamiltonian_from_wavefunction: DTN polynomial Jastrow attached (use_u=%s, use_chi=%s, use_f=%s, C=%d).",
            jastrow_dtn_data.use_u,
            jastrow_dtn_data.use_chi,
            jastrow_dtn_data.use_f,
            jastrow_dtn_data.cutoff_smoothness,
        )
    return hamiltonian_data


def build_hamiltonian_from_projection(
    qe_reference: QEReference,
    projection_result: ProjectionResult,
    *,
    kind: str,
    j1_param: float | None = None,
    j1_type: str = "exp",
    j2_param: float | None = 1.0,
    j2_type: str = "pade",
    ewald_tol: float = 1.0e-10,
    ecp_z_cores: tuple[float, ...] | None = None,
    ecp_by_element: dict[str, ECPData] | None = None,
    jastrow_three_body_data: Jastrow_three_body_data | None = None,
    jastrow_dtn_data: Jastrow_DTN_data | None = None,
    qe_reference_dn: QEReference | None = None,
    projection_result_dn: ProjectionResult | None = None,
    force_complex_gamma: bool = False,
    orthonormalize: bool = True,
) -> Hamiltonian_data:
    """One-shot QE projection -> full :class:`Hamiltonian_data` pipeline.

    Convenience wrapper composing :func:`build_wavefunction_from_projection`
    and :func:`build_hamiltonian_from_wavefunction`. See those two for the
    individual stages; the Jastrow / Ewald arguments are forwarded, as is the
    optional LSDA down-spin channel (``qe_reference_dn`` /
    ``projection_result_dn``). ``force_complex_gamma`` is forwarded to the
    wavefunction builder and is used by the multi-twist writer only.
    ``orthonormalize`` (pure ``kind="pgto"`` band-order MO orthonormalization,
    default ``True``) is forwarded to :func:`build_wavefunction_from_projection`.

    ``jastrow_three_body_data`` and ``jastrow_dtn_data`` are pre-built by the
    caller (the CLI layer holds the basis / orbital choices needed to
    construct them); pass ``None`` to skip the corresponding term.
    """
    structure_data, aos_data, geminal_data = build_wavefunction_from_projection(
        qe_reference,
        projection_result,
        kind=kind,
        qe_reference_dn=qe_reference_dn,
        projection_result_dn=projection_result_dn,
        force_complex_gamma=force_complex_gamma,
        orthonormalize=orthonormalize,
    )
    return build_hamiltonian_from_wavefunction(
        structure_data,
        aos_data,
        geminal_data,
        j1_param=j1_param,
        j1_type=j1_type,
        j2_param=j2_param,
        j2_type=j2_type,
        ewald_tol=ewald_tol,
        ecp_z_cores=ecp_z_cores,
        ecp_by_element=ecp_by_element,
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_dtn_data=jastrow_dtn_data,
    )


# ---------------------------------------------------------------------------
# Multi-twist QE -> combined HDF5 converter
# ---------------------------------------------------------------------------
def convert_qe_to_twist_average_input(
    qe_save_dir: str,
    output_path: str,
    *,
    upf_paths: list[str] | dict[str, str],
    kind: str = "pgto",
    svd_rcond: float | None = None,
    alpha_min: float | None = None,
    exp_tail_eps: float | None = None,
    ecp_by_element: dict[str, ECPData] | None = None,
    ecp_z_cores: tuple[float, ...] | list[float] | None = None,
    j1_param: float | None = None,
    j1_type: str = "exp",
    j2_param: float | None = 1.0,
    j2_type: str = "pade",
    j3_basis_type: str | None = None,
    jastrow_dtn: str | None = None,
    dtn_c: int = 4,
    dtn_n_u: int = 8,
    dtn_n_chi: int = 8,
    dtn_n_eN: int = 4,
    dtn_n_ee: int = 4,
    dtn_cutoff_ratio_u: float = DTN_DEFAULT_CUTOFF_RATIO,
    dtn_cutoff_ratio_chi: float = DTN_DEFAULT_CUTOFF_RATIO,
    dtn_cutoff_ratio_f: float = DTN_DEFAULT_CUTOFF_RATIO,
    ewald_tol: float = 1.0e-10,
    g_cut: float | None = None,
    ecutwfc: str | float | None = None,
    mesh_factor: float = 0.75,
    spline_order: int = 5,
    orthonormalize: bool = True,
) -> None:
    """End-to-end multi-twist QE -> combined HDF5 conversion.

    A multi-k QE
    save (``wfc1.hdf5`` ... ``wfcN.hdf5``, ``data-file-schema.xml``) is
    walked one k-point at a time. For each twist the per-k pGTO
    projection is run, occupation-selected ``MO_up``/``MO_dn`` blocks are
    captured. For restricted inputs, the up-spin twist is ``+k`` and the
    down-spin twist is ``-k``, with ``MO_dn = conj(MO_up)``; the orbitals
    form a time-reversal pair. The aggregated list of per-twist
    ``(k_point_fractional, weight, MO_up, MO_dn)`` is written
    to ``output_path`` via
    :func:`jqmc._twist_average.save_twist_average_input` with
    ``/_meta`` + ``/shared`` + ``/twists``.

    The shared Jastrow template is built from the ``ik=1`` Hamiltonian
    (any k yields the same Jastrow + Coulomb / ECP / structure block
    since those are k-independent); the per-twist HDF5 datasets override
    the geminal block at load time via
    :func:`jqmc._twist_average.load_twist`.

    LSDA (``nspin = 2``) saves are supported for every kind: the layout
    stores the two spin channels' MO blocks separately, and the
    unrestricted template carries ``time_reversal = False`` plus the
    per-spin electron counts. The beta bands stay independent of the alpha
    bands. Each twist stores its ``+k`` point, the up rows (the alpha bands
    at ``+k``) and the down rows ``conj(C_beta)`` (the conjugate of the beta
    bands at ``+k``). The loaded Hamiltonian uses ``+k`` for the up channel
    and ``-k`` for the down channel (:func:`jqmc._twist_average.load_twist`).
    The writer records ``/_meta/spin_twist_pairing = "opposite"``; the loader
    refuses an independent-spin complex-twist input without it, and a
    geminal whose down channel is not at ``-k`` of its up channel fails
    ``sanity_check``.
    Kinds with a spin-independent per-twist
    child ('pgto' / 'pw' / 'pgto+pw') reuse the spin-shared
    ``composite_child``; the bspline kinds additionally emit the
    beta-channel spline payload under ``composite_child_dn`` (the
    per-band control points encode the bands, hence spin-dependent).
    Every twist must fill the SAME integer ``(N_up, N_dn)`` with binary
    band occupations (insulator at fixed magnetization) -- a metallic
    twist raises.

    Args:
        qe_save_dir: Path to the QE ``*.save`` directory or its parent.
        output_path: Destination combined HDF5 (``hamiltonian_data.h5``).
        upf_paths: Element -> UPF path mapping (e.g.
            ``{"Cu": "/abs/path/Cu.ccECP-soft.upf"}`` or a list of
            ``"element=path"`` strings matching the CLI convention), one
            entry per species. It is not auto-discovered here (the
            ``jqmc-tool qe`` / ``run_qe2jqmc`` route does that); a map missing
            a species, or an empty map (without ``ecp_by_element``) on a save
            whose electron count shows removed core electrons, raises
            ``ValueError``.
        kind: Projection kind. Supported: ``"pgto"`` (k-independent AO
            basis, shared across all twists via ``/shared``), ``"pgto+pw"``
            (composite with per-twist complex PW child), ``"pgto+bspline"``
            (composite with per-twist complex B-spline child), and the
            single-basis kinds ``"pw"`` (``project_pw_to_pw``) and
            ``"bspline"`` (``project_pw_to_spline``) whose whole AO leaf is
            the per-twist child. The per-twist child is encoded under
            ``/twists/twist_*/composite_child/``; the shared pGTO basis
            stays in ``/shared``.
        svd_rcond: Canonical-orthogonalisation eigenvalue floor forwarded
            to the per-k projection helper (``project_pw_to_pgto`` for
            ``kind="pgto"``, ``project_pw_to_mixed`` for ``kind="pgto+pw"``,
            ``project_pw_to_pgto_spline`` for ``kind="pgto+bspline"``) and
            to the twist-shared exponent optimisation. ``None`` uses
            ``PW_PGTO_SVD_RCOND``.
        alpha_min: Diffuse-pGTO exponent floor forwarded to the same
            pGTO-bearing projection helpers (``None`` keeps the helper's
            default floor; ``0`` for ``kind="pgto+bspline"``).
        exp_tail_eps: Exponent-tail bound forwarded to the twist-shared
            ``optimize_pgto_exponents`` calls of ``kind="pgto+pw"``
            (``None`` disables it).
        g_cut: For ``kind="pgto+pw"``: explicit PW handoff cutoff in
            ``Bohr^-1``. When ``None`` and ``ecutwfc`` is also ``None``,
            defaults to ``0.5 * sqrt(2 * ecutwfc_k)`` of the QE
            reference at each twist (the same convention as the single-k
            ``run_qe2jqmc`` path). Mutually exclusive with ``ecutwfc``.
            For ``kind="pw"``: the ``|G+k| <= g_cut`` truncation of the PW
            basis (``None`` keeps every plane wave). Ignored for other
            kinds.
        ecutwfc: For ``kind="pgto+pw"``: alternative to ``g_cut`` with
            the same semantics as single-k ``run_qe2jqmc``.

            * ``None`` (default): defer to ``g_cut`` (or its per-twist
              default).
            * Numeric Rydberg value: fixes ``g_cut = sqrt(ecutwfc_Ry)``
              across every twist (matches single-k ``run_qe2jqmc``'s
              numeric mapping).
            * ``"auto"``: per-twist auto-sweep via
              :func:`jqmc.pw_pgto_projection.optimize_mixed_basis`. Each
              twist independently picks the smallest ``g_cut`` whose
              residual meets ``PW_PGTO_AUTO_RESIDUAL_TARGET`` within the
              ``PW_PGTO_AUTO_MAX_ECUTWFC_RY`` ceiling. A
              ``RuntimeError`` is raised on the first twist that fails
              to meet the target (the auto cap is not a per-twist
              "best effort": every twist must clear the same bar so the
              composite has uniform projection quality).

            Mutually exclusive with ``g_cut``; ignored for other kinds.
        mesh_factor: For ``kind="bspline"`` / ``"pgto+bspline"``: spline
            grid size relative to the QE FFT grid, forwarded to
            ``project_pw_to_spline`` / ``project_pw_to_pgto_spline``.
            Default ``0.75``. Ignored for other kinds.
        spline_order: For ``kind="bspline"`` / ``"pgto+bspline"``: spline
            degree (5 = quintic, 3 = cubic), forwarded like
            ``mesh_factor``. Default ``5``. Ignored for other kinds.
        ecp_by_element: Optional analytic ECP records keyed by species
            label or element (case-insensitive), used for the shared
            template build. ECPs parsed from the supplied UPFs take
            precedence.
        ecp_z_cores: Optional per-atom core charges for the shared template
            build. When omitted they are derived from the supplied UPFs (or
            from ``ecp_by_element``); an explicit sequence must contain one
            entry per atom and also feeds DTN's effective charges.
        j1_param: One-body Jastrow parameter of the shared template;
            ``None`` disables J1. Forwarded to the Hamiltonian build.
        j1_type: One-body functional form (``"exp"`` or ``"pade"``).
        j2_param: Two-body Jastrow parameter of the shared template;
            ``None`` disables J2. Default ``1.0``.
        j2_type: Two-body functional form (``"pade"`` or ``"exp"``).
        ewald_tol: Convergence tolerance of the shared template's Ewald
            tables (:py:func:`build_ewald_data`).
        j3_basis_type: Optional AO J3 choice (``"ao"``, ``"ao-small"``,
            ``"ao-medium"``, ``"ao-large"`` or ``"ao-full"``; ``None`` /
            ``"none"`` disables J3).  As for tiled conversion, the J3 basis is
            an independent ordinary periodic Cartesian pGTO basis built from
            the twist-shared UPF shell seeds (after any twist-shared exponent
            optimisation), not the determinant's Bloch AO container.  It is
            phase-free, attached once to the shared template and therefore
            identical for every twist.  ``"mo"`` is rejected: MO J3 orbitals
            are twist-dependent Bloch MOs.
        jastrow_dtn: Optional DTN (Drummond-Towler-Needs) polynomial-Jastrow
            preset (``"u"``, ``"u_chi"``, ``"u_f"``, ``"u_chi_f"``; ``None``
            / ``"none"`` disables it).  DTN is twist-independent and is
            attached once to the shared template.  As on the single-k
            converter, it is mutually exclusive with the Pade J1/J2 family
            and with J3.
        dtn_c: DTN cutoff polynomial smoothness order C (>= 3).
        dtn_n_u: DTN u-term polynomial order N_u.
        dtn_n_chi: DTN chi-term polynomial order N_chi.
        dtn_n_eN: DTN f-term electron-ion polynomial order N_eN.
        dtn_n_ee: DTN f-term electron-electron polynomial order N_ee.
        dtn_cutoff_ratio_u: PBC u cutoff ratio relative to the Wigner-Seitz
            inscribed radius R_ins.
        dtn_cutoff_ratio_chi: PBC chi cutoff ratio relative to R_ins.
        dtn_cutoff_ratio_f: PBC f cutoff ratio relative to R_ins/2.
        orthonormalize: For ``kind="pgto"``: orthonormalize every per-twist
            (and per-spin) MO block in that twist's periodic AO metric by
            band-order Cholesky (see :func:`build_wavefunction_from_projection`).
            Default ``True``; ignored for the other kinds.

    Raises:
        ValueError: If ``kind`` is not one of the supported set.
    """
    _MULTITWIST_SUPPORTED_KINDS = ("pgto", "pw", "bspline", "pgto+pw", "pgto+bspline")
    if kind not in _MULTITWIST_SUPPORTED_KINDS:
        raise ValueError(
            f"convert_qe_to_twist_average_input: kind={kind!r} not supported. Supported kinds: {_MULTITWIST_SUPPORTED_KINDS}."
        )
    _needs_pgto_shells = kind in ("pgto", "pgto+pw", "pgto+bspline")
    # Validate before any source I/O; jqmc_tool is imported lazily as in run_qe2jqmc.
    from .jqmc_tool import build_supercell_jastrow_three_body, resolve_twist_average_j3_choice

    j3_choice = resolve_twist_average_j3_choice(j3_basis_type)

    from .atomic_orbital import AOs_bspline_complex_data, AOs_plane_wave_complex_data
    from ._setting import (
        PW_PGTO_AUTO_MAX_ECUTWFC_RY,
        PW_PGTO_AUTO_RESIDUAL_TARGET,
        PW_PGTO_SVD_RCOND,
    )
    from .pw_pgto_projection import (
        PW_TAPER_KINDS,
        alpha_min_from_gcut,
        filter_shells_by_alpha_min,
        optimize_mixed_basis,
        project_pw_to_mixed,
        project_pw_to_pgto,
        project_pw_to_pgto_spline,
        project_pw_to_pw,
        project_pw_to_spline,
        taper_qe_reference,
    )
    from .pgto_exponent_optimizer import optimize_pgto_exponents
    from .qe_wrapper import _find_save_dir, load_qe_kpoint_list, load_qe_reference
    from .upf_wrapper import UPFExponentInit

    # ecutwfc parsing mirrors single-k ``run_qe2jqmc`` (jqmc_qe.py):
    # ``ecutwfc='auto'`` triggers a per-twist auto-sweep; a numeric
    # value (Rydberg) maps to ``g_cut = sqrt(ecutwfc_Ry)`` shared
    # across twists; ``None`` defers to ``g_cut`` (or its per-twist
    # default).
    _auto_pw_cutoff = False
    _g_cut_from_ecutwfc: float | None = None
    if ecutwfc is not None:
        if g_cut is not None:
            raise ValueError(
                "convert_qe_to_twist_average_input: ``ecutwfc`` and ``g_cut`` are "
                "mutually exclusive (both specify the PW handoff cutoff)."
            )
        ec_str = str(ecutwfc).strip()
        if ec_str.lower() == "auto":
            if kind != "pgto+pw":
                raise ValueError(
                    f"convert_qe_to_twist_average_input: ecutwfc='auto' is only "
                    f"meaningful for kind='pgto+pw' (got kind={kind!r})."
                )
            _auto_pw_cutoff = True
        else:
            try:
                ec_val = float(ec_str)
            except ValueError as exc:
                raise ValueError(
                    f"convert_qe_to_twist_average_input: ecutwfc must be a number in Rydberg or 'auto', got {ec_str!r}."
                ) from exc
            if ec_val < 0.0:
                raise ValueError("convert_qe_to_twist_average_input: ecutwfc must be non-negative (Rydberg).")
            _g_cut_from_ecutwfc = float(np.sqrt(ec_val))

    if svd_rcond is None:
        svd_rcond = PW_PGTO_SVD_RCOND

    # Pin ``occ_only=False`` for the per-k
    # projection so every twist returns the same number of bands
    # (``ref.wfc.n_bands``). ``occ_only=True`` would let twists with
    # different occupation patterns (high-symmetry vs general-k) emit
    # different intermediate basis sizes. This matters for composite
    # per-band AO children even though canonical SD output below always
    # selects exactly ``ne_up`` / ``ne_dn`` MO rows and persists no lambda.
    _occ_only_for_multitwist = False

    # Resolve UPF map (accept either list or dict; matches CLI ergonomics).
    if isinstance(upf_paths, dict):
        upf_map = dict(upf_paths)
    else:
        upf_map = {}
        for spec in upf_paths:
            if "=" not in spec:
                raise ValueError(f"upf_paths entry expects 'element=path.upf', got {spec!r}.")
            elem, path = spec.split("=", 1)
            upf_map[elem.strip()] = path.strip()

    # 1) Enumerate twists from the XML (weights only -- per-k loading
    #    happens below). This also drives the wfcN.hdf5 file count.
    save_dir = _find_save_dir(qe_save_dir)
    xml_path = os.path.join(save_dir, "data-file-schema.xml")
    from .qe_wrapper import load_qe_lsda

    # LSDA (nspin=2): the combined twist layout stores the two spin channels'
    # MO blocks separately (``mo_coefficients_up`` / ``mo_coefficients_dn``
    # per twist), so an *insulating* spin-polarized save (binary
    # occupations at every twist, fixed magnetization) converts through
    # the same machinery. Child handling per kind:
    #   * PW child (G-index list): spin-independent -- same k-sphere and
    #     same g_cut for both spins -- so the spin-shared
    #     ``composite_child`` covers 'pw' / 'pgto+pw'.
    #   * Spline child: the per-band control points encode the bands
    #     themselves (spin-dependent), so 'bspline' / 'pgto+bspline'
    #     additionally emit the beta-channel payload under
    #     ``composite_child_dn`` (the loader flips it to -k, following the
    #     BCS / TR-pair convention).
    _is_lsda = load_qe_lsda(xml_path)
    kpts_meta = load_qe_kpoint_list(xml_path)
    n_twist = len(kpts_meta)
    logger.info("QE multi-twist convert: %d k-points found in %s", n_twist, xml_path)

    # 2) Build the shared template Hamiltonian from ik=1 (k-independent
    #    fields: structure, Coulomb / Ewald, Jastrow, AO basis shape).
    if _is_lsda:
        ref1 = load_qe_reference(qe_save_dir, ik=1, spin="up")
        ref1_dn = load_qe_reference(qe_save_dir, ik=1, spin="dw")
    else:
        ref1 = load_qe_reference(qe_save_dir, ik=1)
        ref1_dn = None
    # Vacuum cells only (bulk stays bit-identical): taper the PW coefficients
    # at the G-sphere edge before any plane-wave-carrying projection. The
    # decision depends on the geometry alone, so it is the same for every
    # twist; the ik=1 metadata is what the file records.
    pw_taper_meta = None
    # A truncated ``pw`` basis keeps only |G+k| <= g_cut, so its edge is g_cut.
    _taper_edge = float(g_cut) if (kind == "pw" and g_cut is not None) else None
    if kind in PW_TAPER_KINDS:
        ref1, pw_taper_meta = taper_qe_reference(ref1, g_end=_taper_edge, label="_ik1")
        if ref1_dn is not None:
            ref1_dn, _ = taper_qe_reference(ref1_dn, g_end=_taper_edge, label="_ik1_dn")
    g_target = float(np.sqrt(2.0 * ref1.ecutwfc))
    # Parse UPFs for every kind, not only the pGTO-bearing ones: raw PW and
    # B-spline conversions still need the analytic ECP and its z_core for the
    # Coulomb potential and DTN effective charges.
    strategy = None
    if upf_map:
        _require_upf_for_every_species(upf_map, list(ref1.atom_labels), list(ref1.atom_elements))
        cell_min_edge = float(np.min(np.linalg.norm(ref1.cell.a, axis=1)))
        strategy = UPFExponentInit(
            upf_map,
            g_target=g_target,
            cell_min_edge=cell_min_edge,
        )

    # The pGTO shells are needed only for kinds that carry a pGTO block
    # (kind='pgto' / 'pgto+pw' / 'pgto+bspline'). Single-basis kinds
    # ('pw' / 'bspline') skip the UPF fit; ``upf_paths`` is still honoured
    # for ``ecp_by_element`` lookup but the shells loop is a no-op.
    if _needs_pgto_shells:
        if strategy is None:
            raise ValueError(f"kind={kind!r} requires upf_paths=['element=path.upf', ...] to seed the pGTO basis.")
        # Cell-aware diffuse floor for the AE bare-Coulomb pgto ladder
        # (PBC ill-conditioning guard, see
        # :func:`jqmc.upf_wrapper.generate_ae_even_tempered_shells`).
        shells = [
            strategy.shells_for_element(_upf_species_key(strategy, label, element))
            for label, element in zip(ref1.atom_labels, ref1.atom_elements, strict=True)
        ]
    else:
        shells = None

    # Resolve one shared exponent set before entering the twist loop.  The
    # pGTO child is stored once under /shared, so optimising independently at
    # each k would be both semantically invalid and 64x more expensive.  ik=1
    # is the calibration target; every later twist is still projected and its
    # residual checked by the normal auto/fixed path using this shared basis.
    if kind == "pgto+pw" and _auto_pw_cutoff:
        seed_shells = shells
        opt_refs = [ref1] if ref1_dn is None else [ref1, ref1_dn]
        auto_max_g_cut = float(np.sqrt(PW_PGTO_AUTO_MAX_ECUTWFC_RY))
        user_fixed_alpha = None if alpha_min is None else float(alpha_min)
        seed_sweep = optimize_mixed_basis(
            ref1,
            seed_shells,
            residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
            drop_diffuse=True,
            fixed_alpha_min=user_fixed_alpha,
            max_g_cut=auto_max_g_cut,
            svd_rcond=svd_rcond,
            occ_only=_occ_only_for_multitwist,
        )
        for seed_trial in seed_sweep.trials:
            if seed_trial.mean_residual <= PW_PGTO_AUTO_RESIDUAL_TARGET:
                break
            trial_floor = float(seed_trial.alpha_min if user_fixed_alpha is None else user_fixed_alpha)
            active_shells = filter_shells_by_alpha_min(seed_shells, trial_floor)
            if not any(atom_shells for atom_shells in active_shells):
                continue
            exponent_opt = optimize_pgto_exponents(
                opt_refs,
                active_shells,
                occ_only=True,
                svd_rcond=svd_rcond,
                g_cut=float(seed_trial.g_cut),
                exponent_floor=trial_floor,
                exp_tail_eps=exp_tail_eps,
            )
            if not exponent_opt.accepted:
                continue
            candidate = optimize_mixed_basis(
                ref1,
                exponent_opt.shells_per_atom,
                residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
                g_cut_values=[float(seed_trial.g_cut)],
                fixed_alpha_min=trial_floor,
                svd_rcond=svd_rcond,
                occ_only=_occ_only_for_multitwist,
            )
            if candidate.target_met:
                shells = exponent_opt.shells_per_atom
                logger.info(
                    "Twist-shared pGTO exponent optimisation lowered the ik=1 "
                    "auto cutoff from %.3f to %.3f Ry; mean residual %.3e.",
                    seed_sweep.best.g_cut**2,
                    candidate.best.g_cut**2,
                    candidate.best.mean_residual,
                )
                break

    fixed_exponent_g_cut = None
    if kind == "pgto+pw" and not _auto_pw_cutoff:
        if _g_cut_from_ecutwfc is not None:
            fixed_exponent_g_cut = float(_g_cut_from_ecutwfc)
        elif g_cut is not None:
            fixed_exponent_g_cut = float(g_cut)
    if fixed_exponent_g_cut is not None:
        fixed_floor = float(alpha_min) if alpha_min is not None else alpha_min_from_gcut(fixed_exponent_g_cut)
        active_shells = filter_shells_by_alpha_min(shells, fixed_floor)
        if any(atom_shells for atom_shells in active_shells):
            opt_refs = [ref1] if ref1_dn is None else [ref1, ref1_dn]
            exponent_opt = optimize_pgto_exponents(
                opt_refs,
                active_shells,
                occ_only=True,
                svd_rcond=svd_rcond,
                g_cut=fixed_exponent_g_cut,
                exponent_floor=fixed_floor,
                # Keep the spilling search WFC-bounded; its condition
                # constraint is independently evaluated on the raw full
                # overlap of the emitted AO basis.
                fixed_mixed_overlap=False,
                deployed_fixed_mixed_overlap=True,
                exp_tail_eps=exp_tail_eps,
            )
            shells = exponent_opt.shells_per_atom
            logger.info(
                "Twist-shared fixed pgto+pw exponent optimisation at %.3f Ry: "
                "accepted=%s, occupation-weighted spilling %.6e -> %.6e.",
                fixed_exponent_g_cut**2,
                exponent_opt.accepted,
                exponent_opt.deployed_initial_spilling,
                exponent_opt.deployed_optimized_spilling,
            )

    # Per-k projection dispatch. The composite-kind branches return a
    # projection that carries both the shared pgto block (k-independent
    # primitives) and the k-dependent child (PW G-list / spline cube).
    # ``g_cut`` for ``pgto+pw`` mirrors single-k ``run_qe2jqmc``'s
    # default (``0.5 * sqrt(2 * ecutwfc)``); explicit override flows
    # through verbatim.
    def _project_one(
        ref_k: QEReference,
        g_cut_fixed: float | None = None,
    ) -> ProjectionResult:
        if kind == "pgto":
            return project_pw_to_pgto(
                ref_k,
                shells,
                occ_only=_occ_only_for_multitwist,
                svd_rcond=svd_rcond,
                alpha_min=alpha_min,
            )
        if kind == "pw":
            g_cut_val = float(g_cut) if g_cut is not None else None
            return project_pw_to_pw(ref_k, g_cut=g_cut_val, occ_only=_occ_only_for_multitwist)
        if kind == "bspline":
            return project_pw_to_spline(
                ref_k,
                mesh_factor=mesh_factor,
                order=spline_order,
                occ_only=_occ_only_for_multitwist,
            )
        if kind == "pgto+pw":
            # Per-k auto sweep via optimize_mixed_basis (mirrors
            # single-k ``ecutwfc='auto'``). Each twist must
            # independently meet the residual target within the auto
            # cap; requiring every twist to clear the same bar keeps
            # the composite's projection quality uniform across the BZ.
            # If any twist fails, the converter raises (no silent
            # fallback to a coarser g_cut).
            # ``g_cut_fixed`` (the LSDA dn channel) bypasses the sweep so
            # both spin blocks share the up-channel's PW / pGTO split.
            if _auto_pw_cutoff and g_cut_fixed is None:
                auto_max_g_cut = float(np.sqrt(PW_PGTO_AUTO_MAX_ECUTWFC_RY))
                user_fixed_alpha = None if alpha_min is None else float(alpha_min)
                sweep = optimize_mixed_basis(
                    ref_k,
                    shells,
                    residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
                    drop_diffuse=True,
                    fixed_alpha_min=user_fixed_alpha,
                    max_g_cut=auto_max_g_cut,
                    svd_rcond=svd_rcond,
                    # Pin the multi-twist all-bands policy (see
                    # ``_occ_only_for_multitwist`` above): the sweep's best
                    # projection is returned as-is, so it must carry the
                    # same uniform band count as every other branch.
                    occ_only=_occ_only_for_multitwist,
                )
                if not sweep.target_met:
                    raise RuntimeError(
                        f"convert_qe_to_twist_average_input(ecutwfc='auto'): per-twist "
                        f"residual did not reach the target "
                        f"{PW_PGTO_AUTO_RESIDUAL_TARGET:.2e} within the auto cap "
                        f"({PW_PGTO_AUTO_MAX_ECUTWFC_RY:.1f} Ry); achieved mean residual "
                        f"{sweep.best.mean_residual:.3e}. Re-run with a numeric "
                        f"``ecutwfc`` (Rydberg) larger than {PW_PGTO_AUTO_MAX_ECUTWFC_RY:.1f}, "
                        f"or use ``g_cut`` to override the per-twist handoff explicitly."
                    )
                logger.info(
                    "Auto pgto+pw twist: target=%.2e MET, g_cut=%.3f Bohr^-1 (ecutwfc=%.3f Ry), n_pw=%d, mean residual=%.3e.",
                    PW_PGTO_AUTO_RESIDUAL_TARGET,
                    sweep.best.g_cut,
                    sweep.best.g_cut**2,
                    sweep.best.n_pw,
                    sweep.best.mean_residual,
                )
                return sweep.best
            g_max = float(np.sqrt(2.0 * ref_k.ecutwfc))
            if g_cut_fixed is not None:
                g_cut_val = float(g_cut_fixed)
            elif g_cut is not None:
                g_cut_val = float(g_cut)
            elif _g_cut_from_ecutwfc is not None:
                g_cut_val = _g_cut_from_ecutwfc
            else:
                g_cut_val = 0.5 * g_max
            return project_pw_to_mixed(
                ref_k,
                shells,
                g_cut=g_cut_val,
                drop_diffuse=True,
                occ_only=_occ_only_for_multitwist,
                svd_rcond=svd_rcond,
                alpha_min=alpha_min,
            )
        if kind == "pgto+bspline":
            return project_pw_to_pgto_spline(
                ref_k,
                shells,
                mesh_factor=mesh_factor,
                order=spline_order,
                alpha_min=(0.0 if alpha_min is None else float(alpha_min)),
                svd_rcond=svd_rcond,
                occ_only=_occ_only_for_multitwist,
            )
        raise ValueError(f"Unreachable: kind={kind!r} passed the guard but no projection branch.")

    def _project_pair(
        ref_up_k: QEReference,
        ref_dn_k: QEReference | None,
    ) -> tuple[ProjectionResult, ProjectionResult | None]:
        """Project one twist; for LSDA also the dn channel (shared g_cut)."""
        proj_up_k = _project_one(ref_up_k)
        if ref_dn_k is None:
            return proj_up_k, None
        _g_fix = float(proj_up_k.g_cut) if (kind == "pgto+pw" and _auto_pw_cutoff) else None
        proj_dn_k = _project_one(ref_dn_k, g_cut_fixed=_g_fix)
        if kind in ("pw", "pgto+pw") and not np.array_equal(np.asarray(proj_up_k.pw_millers), np.asarray(proj_dn_k.pw_millers)):
            raise RuntimeError(
                "Internal: LSDA up/dn PW G-lists diverged at a twist; the "
                "spin-shared composite child of the combined layout would be invalid."
            )
        return proj_up_k, proj_dn_k

    proj1, proj1_dn = _project_pair(ref1, ref1_dn)

    # Resolve per-atom core charges and analytic ECP blocks exactly as the
    # single-k TOML path does.  The derived z_core is also the source of DTN's
    # z_eff = Z - z_core; omitting this on a B-spline-only conversion would
    # incorrectly give Li.ccECP z_eff=3 instead of 1.
    structure1, _ = _build_structure_data(ref1)
    resolved_ecp_by_element: dict[str, ECPData] | None = None
    if ecp_by_element:
        supplied_ecp = {label.lower(): ecp for label, ecp in ecp_by_element.items()}
        resolved_ecp_by_element = {}
        for atom_label, atom_element in zip(ref1.atom_labels, ref1.atom_elements, strict=True):
            ecp = supplied_ecp.get(atom_label.lower()) or supplied_ecp.get(atom_element.lower())
            if ecp is not None:
                resolved_ecp_by_element[atom_label] = ecp
        if not resolved_ecp_by_element:
            resolved_ecp_by_element = None
    z_cores_per_atom: tuple[float, ...] | None = None
    if ecp_z_cores is not None:
        z_core_values = [float(z) for z in ecp_z_cores]
        if len(z_core_values) != int(structure1.natom):
            raise ValueError(
                f"ecp_z_cores has {len(z_core_values)} entries but the QE structure has {int(structure1.natom)} atoms."
            )
        z_cores_per_atom = tuple(z_core_values)
    else:
        z_core_values = []
        for atom_label, atom_element, atomic_number in zip(
            ref1.atom_labels,
            ref1.atom_elements,
            structure1.atomic_numbers,
            strict=True,
        ):
            z_core = 0.0
            if strategy is not None:
                key = _upf_species_key(strategy, atom_label, atom_element)
                z_core = float(atomic_number) - strategy.z_valence_for_element(key)
            elif resolved_ecp_by_element is not None:
                ecp = resolved_ecp_by_element.get(atom_label) or resolved_ecp_by_element.get(atom_element)
                if ecp is not None:
                    z_core = float(ecp.z_core)
            z_core_values.append(z_core)
    if strategy is not None:
        parsed_ecp_by_element: dict[str, ECPData] = {}
        for atom_label, atom_element in zip(ref1.atom_labels, ref1.atom_elements, strict=True):
            key = _upf_species_key(strategy, atom_label, atom_element)
            if not strategy.is_coulomb_ae_for_element(key) and atom_label not in parsed_ecp_by_element:
                parsed_ecp_by_element[atom_label] = strategy.ecp_for_element(key)
        if parsed_ecp_by_element:
            resolved_ecp_by_element = parsed_ecp_by_element
            if z_cores_per_atom is None:
                z_cores_per_atom = tuple(z_core_values)
    elif any(z > 0.0 for z in z_core_values) and resolved_ecp_by_element is None:
        raise ValueError("ECP core charges were detected but no UPF/ECP source is available.")

    dtn_coulomb_potential_data, _ = _build_qe_coulomb_potential_data(
        structure1,
        int(round(float(ref1.bands.n_electrons))),
        z_cores_per_atom,
        resolved_ecp_by_element,
    )

    # One independent phase-free periodic AO J3 for the physical cell, shared
    # by every twist (the untiled analogue of the tiled converter's
    # supercell J3). Its shells are the twist-shared seeds, never a twist's
    # Bloch AO leaf, so it exists for every projection kind.
    jastrow_three_body_data = None
    if j3_choice is not None:
        if strategy is None:
            raise ValueError(f"j3_basis_type={j3_choice!r} requires upf_paths=['element=path.upf', ...] to seed the J3 basis.")
        j3_shells = shells
        if j3_shells is None:
            j3_shells = [
                strategy.shells_for_element(_upf_species_key(strategy, label, element))
                for label, element in zip(ref1.atom_labels, ref1.atom_elements, strict=True)
            ]
        jastrow_three_body_data = build_supercell_jastrow_three_body(structure1, j3_shells, j3_choice)

    jastrow_dtn_data = build_jastrow_dtn_from_options(
        structure1,
        jastrow_dtn=jastrow_dtn,
        coulomb_potential_data=dtn_coulomb_potential_data,
        dtn_c=dtn_c,
        dtn_n_u=dtn_n_u,
        dtn_n_chi=dtn_n_chi,
        dtn_n_eN=dtn_n_eN,
        dtn_n_ee=dtn_n_ee,
        dtn_cutoff_ratio_u=dtn_cutoff_ratio_u,
        dtn_cutoff_ratio_chi=dtn_cutoff_ratio_chi,
        dtn_cutoff_ratio_f=dtn_cutoff_ratio_f,
    )
    template_ham = build_hamiltonian_from_projection(
        ref1,
        proj1,
        kind=kind,
        j1_param=j1_param,
        j1_type=j1_type,
        j2_param=j2_param,
        j2_type=j2_type,
        ewald_tol=ewald_tol,
        ecp_z_cores=z_cores_per_atom,
        ecp_by_element=resolved_ecp_by_element,
        jastrow_three_body_data=jastrow_three_body_data,
        jastrow_dtn_data=jastrow_dtn_data,
        qe_reference_dn=ref1_dn,
        projection_result_dn=proj1_dn,
        force_complex_gamma=True,
        orthonormalize=orthonormalize,
    )

    # 3) Walk every twist; collect per-k occupied-MO blocks. The per-twist
    #    fractional k is read from the AO leaf (``orb_data_up_spin.aos_data
    #    .k_point``) since the structure carries no k. ``save_twist_average_input``
    #    consumes a tuple per twist, with ``None`` reserved for Gamma so the
    #    real-path dispatch stays byte-identical. For composite
    #    kinds we also collect the per-twist k-dependent child of the
    #    composite AO leaf.
    # Streamed: each twist is written to the output HDF5 as soon as it is
    # built, and its arrays are dropped before the next projection starts.
    # Accumulating them instead holds ``n_twist`` B-spline
    # control-point tables at once -- 25.2 GiB EACH at 108 atoms,
    # 501 bands, mesh_factor 0.5, i.e. ~100 GiB for 4 twists on top of the
    # template and the live projection buffers, which runs out of memory
    # partway through twist 2 on a 251 GB node. Only the weights (scalars) are kept.
    from ._twist_average import TwistAverageWriter

    weights: list[float] = []
    _t_gem = template_ham.wavefunction_data.geminal_data
    _writer = TwistAverageWriter(output_path, template_ham, n_twist)
    if pw_taper_meta is not None:
        _writer.record_attrs(pw_taper_meta)
    _emit_child = kind != "pgto"
    try:
        for ik in range(1, n_twist + 1):
            if _is_lsda:
                ref_k = load_qe_reference(qe_save_dir, ik=ik, spin="up")
                ref_dn_k = load_qe_reference(qe_save_dir, ik=ik, spin="dw")
            else:
                ref_k = load_qe_reference(qe_save_dir, ik=ik)
                ref_dn_k = None
            if kind in PW_TAPER_KINDS:
                ref_k, _meta_k = taper_qe_reference(ref_k, g_end=_taper_edge, label=f"_ik{ik}")
                if _meta_k is not None:
                    _writer.record_attrs(_meta_k)
                if ref_dn_k is not None:
                    ref_dn_k, _meta_dn_k = taper_qe_reference(ref_dn_k, g_end=_taper_edge, label=f"_ik{ik}_dn")
                    if _meta_dn_k is not None:
                        _writer.record_attrs(_meta_dn_k)
            proj_k, proj_dn_k = _project_pair(ref_k, ref_dn_k)
            sd_k, aos_k, gem_k = build_wavefunction_from_projection(
                ref_k,
                proj_k,
                kind=kind,
                qe_reference_dn=ref_dn_k,
                projection_result_dn=proj_dn_k,
                force_complex_gamma=True,
                orthonormalize=orthonormalize,
            )
            # The template's (ne_up, ne_dn) are shared by every twist at load
            # time (load_twist reads them from /shared), so the per-twist
            # fillings must be uniform: an insulator at fixed magnetization.
            # A twist with a different filling (metallic pocket / magnetization
            # jump) cannot be represented -- fail instead of averaging
            # inconsistent determinants. (Binary 0/1 occupations per twist are
            # already enforced inside the geminal builder.)
            if (int(gem_k.num_electron_up), int(gem_k.num_electron_dn)) != (
                int(_t_gem.num_electron_up),
                int(_t_gem.num_electron_dn),
            ):
                raise ValueError(
                    f"convert_qe_to_twist_average_input: twist {ik} fills "
                    f"(ne_up, ne_dn) = ({gem_k.num_electron_up}, {gem_k.num_electron_dn}) "
                    f"but the ik=1 template fills ({_t_gem.num_electron_up}, "
                    f"{_t_gem.num_electron_dn}). Twist-averaging needs a uniform "
                    "filling across the BZ (insulator at fixed magnetization)."
                )
            # ``k_point`` lives on the AO leaf, not the structure.
            # For composite leaves (``pgto+pw`` / ``pgto+bspline``) the
            # k-dependent twist sits on the k-dep child (PW or bspline in slot
            # 1; pgto in slot 0 is k-independent). For single-basis (``pw`` /
            # ``bspline`` / ``pgto``) the top leaf carries it directly.
            _aos_top = gem_k.orb_data_up_spin.aos_data
            if isinstance(_aos_top, AOs_composite_data):
                _kp_k = getattr(_aos_top.children[1], "k_point", None)
            else:
                _kp_k = getattr(_aos_top, "k_point", None)
            _k_point_k = None if _kp_k is None else tuple(float(x) for x in _kp_k)
            weights.append(float(kpts_meta[ik - 1][1]))
            _mo_up_k = np.asarray(gem_k.orb_data_up_spin.mo_coefficients)
            _mo_dn_k = np.asarray(gem_k.orb_data_dn_spin.mo_coefficients)
            # Extract the per-twist k-dependent AO leaf state for storage.
            # For kind='pgto' (k-independent basis) there is nothing per-twist.
            # For composite kinds (pgto+pw / pgto+bspline) the k-dep child is
            # ``aos_k.children[1]`` (pgto in slot 0). For single-basis kinds
            # (pw / bspline) the whole AO leaf IS the per-twist k-dep state.
            if kind == "pgto":
                _child_k = None
            else:
                if isinstance(aos_k, AOs_composite_data):
                    if len(aos_k.children) < 2:
                        raise RuntimeError(
                            f"Internal: kind={kind!r} expected a composite AO leaf with >=2 children; "
                            f"got {len(aos_k.children)} children."
                        )
                    k_dep = aos_k.children[1]
                else:
                    k_dep = aos_k
                if isinstance(k_dep, AOs_plane_wave_complex_data):
                    _child_k = ("pw_complex", np.asarray(k_dep._g_indices, dtype=np.int64))
                elif isinstance(k_dep, AOs_bspline_complex_data):
                    _child_k = ("bspline_complex", np.asarray(k_dep.spline_coefficients, dtype=np.complex128))
                else:
                    raise RuntimeError(
                        f"Internal: kind={kind!r} produced an unexpected k-dependent child type "
                        f"{type(k_dep).__name__}; expected AOs_plane_wave_complex_data or "
                        "AOs_bspline_complex_data."
                    )
            # LSDA + spline child: the per-band control points are the beta
            # bands themselves, so the dn channel needs its own payload (the
            # PW G-list child is spin-shared -- equality asserted in
            # ``_project_pair``). Stored in the +k representation; the loader
            # applies the BCS / TR-pair -k flip.
            if _is_lsda and kind in ("bspline", "pgto+bspline"):
                _child_dn_k = ("bspline_complex", np.asarray(proj_dn_k.spline_coefficients, dtype=np.complex128))
            else:
                _child_dn_k = None
            logger.info(
                "QE -> twist %d/%d: xk_bohr_inv=%s, weight=%.6f, MO dtype=%s",
                ik,
                n_twist,
                np.asarray(ref_k.wfc.k_point).tolist(),
                weights[-1],
                str(_mo_up_k.dtype),
            )
            # Write and release: after this the twist's control-point table is on
            # disk and nothing in this frame references it.
            _writer.write_twist(
                ik - 1,
                _k_point_k,
                weights[-1],
                _mo_up_k,
                _mo_dn_k,
                composite_child=_child_k if _emit_child else None,
                composite_child_dn=_child_dn_k,
            )
            del _mo_up_k, _mo_dn_k, _child_k, _child_dn_k
            del ref_k, ref_dn_k, proj_k, proj_dn_k, sd_k, aos_k, gem_k

        # 4) Finalise the combined HDF5 (/shared and every /twists/twist_* group
        #    were already written inside the loop above).
    except BaseException:
        # Close the handle so the partially written file is flushed and
        # inspectable; its /_meta/bz_weight_sum stays 0.0, which the
        # loader rejects, so a crashed conversion can never masquerade
        # as a complete input.
        _writer._f.close()
        raise
    _writer.close()
    logger.info(
        "QE multi-twist convert: wrote combined HDF5 (n_twist=%d, sum_weights=%.6f) -> %s",
        n_twist,
        float(np.sum(weights)),
        output_path,
    )

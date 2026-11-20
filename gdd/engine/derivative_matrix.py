"""Derivative matrix-universe enumeration (GDD engine, requirement side).

Implements the cell-enumeration half of ``derivatives.continuous_fixture_matrix.v1``:
the deterministic pairwise covering array (plus the selective 3-way twist and
periodic-ECP Jastrow blocks) over the declared AO / Jastrow / potential /
boundary-condition / geminal axes, and the declared builder model the greedy
cover uses as a tie-break. Local artifact availability never selects obligations.
This module never imports pytest, jqmc, JAX,
or a test module; Hamiltonian construction and execution stay in
``tests/test_jqmc_kernel/_generated_matrix_base.py``.

``_AO_FAMILY_TOKENS`` and ``_JASTROW_CHANNEL_TOKENS`` are DECLARED contract
vocabularies, not introspection.  The structural gate in
``tests/test_jqmc_kernel/test_gdd_derivatives_contract.py`` verifies that they
together with the specialized frozen-adapter contracts account for production
introspection of ``jqmc.atomic_orbital`` and
``jqmc.jastrow_factor.Jastrow_data``.
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

import hashlib
import itertools
import os
from pathlib import Path

from gdd.engine import derivative_exclusions
from gdd.engine.pinned_fixtures import fixture_exists

_ROOT = Path(__file__).resolve().parents[2]

# Declared general matrix AO leaf families (tokens = class name minus AOs_/_data).
_AO_FAMILY_TOKENS = (
    "bspline",
    "bspline_complex",
    "cart_gto",
    "cart_pgto",
    "cart_pgto_complex",
    "composite",
    "plane_wave",
    "plane_wave_complex",
    "sphe_gto",
    "sphe_pgto",
    "sphe_pgto_complex",
)

# The legacy tiled B-spline adapter stores multiple primitive k blocks, whereas
# a matrix cell owns one Bloch leaf and admits independent geminal parameters.
# Its frozen SD contract is checked separately, not silently omitted from the
# production inventory. The structural gate requires live full/mixed derivative
# evidence for every entry here; the common TiledOrbitalSet is not an AO leaf.
_SPECIALIZED_AO_FAMILY_CONTRACTS = {
    "bspline_tiled_complex": ("atomic_orbital.yaml", "MATH-AO-TILED-BSPLINE-DERIVATIVE-AD-001"),
}
_PRODUCTION_AO_FAMILY_TOKENS = tuple(sorted((*_AO_FAMILY_TOKENS, *_SPECIALIZED_AO_FAMILY_CONTRACTS)))


# Declared Jastrow channels of Jastrow_data (pytree fields), plus the none baseline.
_JASTROW_CHANNEL_TOKENS = ("DTN", "J1", "J2", "J3poly", "none")


def _ao_families() -> list[str]:
    """The general matrix AO leaf-family tokens, sorted (adapters have dedicated contracts)."""
    return list(_AO_FAMILY_TOKENS)


def _jastrow_types() -> list[str]:
    """The declared Jastrow channel tokens, sorted."""
    return list(_JASTROW_CHANNEL_TOKENS)


_GEMS = ("SD", "AGPn", "AGP", "AGP_MO")

_QE_STEM = "QE:Li_gamma_pgto+pw"  # sentinel stem routed to the QE composite builder
_QE_SAVE = str(_ROOT / "tests" / "qe_example_files" / "Li_gamma" / "out" / "li.save")
_QE_AVAILABLE = all(fixture_exists(os.path.join(_QE_SAVE, f)) for f in ("data-file-schema.xml", "wfc1.hdf5", "Li.ccECP.upf"))

_BCS = ("mol", "pbc_gamma", "twist")
_ECPS = ("AE", "ECP")


def _bc_allowed(ao: str) -> set[str]:
    """Physical boundary conditions admissible for an AO leaf family -- MIRRORS the gate's
    assumptions: complex-Bloch leaf => twist; real periodic (pgto/plane_wave/bspline) => PBC;
    composite (pgto(+)PW) => PBC-Gamma OR twist; molecular GTO => mol.
    """
    if "complex" in ao:
        return {"twist"}
    if ao == "composite":
        return {"pbc_gamma", "twist"}
    if "pgto" in ao or "plane_wave" in ao or "bspline" in ao:
        return {"pbc_gamma"}
    return {"mol"}  # cart_gto / sphe_gto


def _agp_dense_constructible(ao: str, bc: str) -> bool:
    """Whether a WELL-CONDITIONED dense-AO-AGP (``gem == AGP``) is constructible for AO family
    ``ao`` under boundary condition ``bc``. The dense-AO-AGP a==b compare is only MEANINGFUL when
    the AO overlap is well-conditioned (full-rank, no image-truncation near-singularity): both the
    analytic derivative and the ``_auto`` twin differentiate the natural-orbital SVD of
    ``f = S_up^{1/2} lambda S_dn^{1/2}``, which is non-differentiable garbage on a near-singular
    overlap. Admitted ONLY on:

      * molecular GTO leaves (``cart_gto`` / ``sphe_gto``, always ``mol``) -- a standard basis
        with a well-conditioned overlap; and
      * the QE pgto(+)PW ``composite`` at BOTH ``pbc_gamma`` (REAL lambda) AND ``twist`` (COMPLEX
        HERMITIAN lambda) -- both well-conditioned BY CONSTRUCTION (windowed pGTO + low-|G| PW; the
        EXACT 8-set ``dense_agp`` basis, overlap cond ~1.6e4, PSD). The complex-twist full-rank lift
        ``_make_full_rank_ao_agp`` keeps the complex dtype with a Hermitian ramp (its ``is_complex``
        branch), so the natural-orbital SVD of the complex Hermitian ``f`` stays differentiable.

    EXCLUDED: periodic bare-pGTO (``sphe_pgto`` / ``cart_pgto`` and their complex-twist leaves)
    -- the lattice-image truncation drives the AO overlap near-singular (min eig < 0), the exact
    condition the 8-set harness deliberately avoids; the flat plane-wave / B-spline leaves
    (no ``MOs_data`` -> ``convert_from_MOs_to_AOs`` fold, and no conditioning guarantee). These are
    excluded for CONDITIONING reasons, unrelated to the real/complex lambda dtype.
    """
    if ao in ("cart_gto", "sphe_gto"):
        return True
    return ao == "composite" and bc in ("pbc_gamma", "twist")


def _split_cell(cell: tuple) -> tuple[str, str, str, str, str]:
    """Unpack a cell tuple, treating a bare 4-tuple ``(ao, jas, ecp, bc)`` as ``gem == SD``
    (so the SD sub-matrix generation reuses the 4-axis helpers byte-identically).
    """
    if len(cell) == 4:
        ao, jas, ecp, bc = cell
        return ao, jas, ecp, bc, "SD"
    return cell


#: AO families whose MO leaves are atom-centred Gaussians. The whitened dense-MO AGP
#: (``gem == AGP_MO``) is admitted only here; see ``_agp_mo_constructible``.
_AGP_MO_AO_FAMILIES = (
    "cart_gto",
    "cart_pgto",
    "cart_pgto_complex",
    "sphe_gto",
    "sphe_pgto",
    "sphe_pgto_complex",
)


def _agp_mo_constructible(ao: str) -> bool:
    """Whether a dense MO-basis AGP (``gem == AGP_MO``) is a MEANINGFUL cell for AO family ``ao``.

    Admitted on the atom-centred Gaussian families only. EXCLUDED: the QE-projection leaves
    (``plane_wave`` / ``bspline`` / ``composite`` and their complex-twist siblings).

    Those leaves' all-electron bands sampled at ion-centred points are near-collinear
    (cond(M) ~ 300-400), which the AGP form squares. The fixture builder answered that by
    WHITENING lambda at the ion reference (``lambda = W T W^H``, ``W = pinv(M0^T)``), which does
    make the cell constructible -- and numerically empty. Measured on the resulting fixtures:

      * ``de_L/dlambda`` peaks at 1e-9 .. 1e-17 against a cell scale of ~1e+02, i.e. the local
        energy does not see lambda at all. Confirmed by THREE independent oracles -- the analytic
        derivative, autodiff, and a finite difference of e_L itself (max |FD| = 1.3e-08) -- so it
        is the fixture, not either implementation. Deleting the analytic lambda derivative left
        those cells GREEN.
      * ``dlnPsi/dlambda`` stays healthy (5.5e+01 .. 3.3e+02) on the same fixtures: lambda acts
        essentially as an overall normalization of Psi, which lnPsi sees and the ratio HPsi/Psi
        is invariant to.
      * the surviving physical axes sit at the edge of fp64 for cond_2(lambda) ~ 5e4: de_L/dr
        agreed to 8e-08 of the gradient magnitude on one machine and 1.2e-06 on another, so any
        tolerance that passes is a tolerance calibrated to one environment.

    A cell that certifies nothing about its own defining parameter is not evidence, and the
    whitening was what made it constructible in the first place. So the honest statement is the
    one that held BEFORE the whitening: on these leaves the dense MO-AGP is not constructible.
    The six Gaussian leaves keep the AGP_MO coverage (lambda sensitivity 1.4e-2 .. 4.6e-2).
    """
    return ao in _AGP_MO_AO_FAMILIES


def _predicate_valid(cell: tuple) -> bool:
    """The engineering predicates above, applied to one cell (the reviewed rationale).

    The generator does not call this directly: ``_valid`` reads the reviewed table in
    ``gdd/engine/derivative_exclusions.py``, and ``scripts/check_gdd_derivative_exclusions.py``
    requires this predicate and the table to exclude exactly the same cells.
    """
    ao, _jas, _ecp, bc, gem = _split_cell(cell)
    if bc not in _bc_allowed(ao):
        return False
    if gem == "AGP" and not _agp_dense_constructible(ao, bc):
        return False
    if gem == "AGP_MO" and not _agp_mo_constructible(ao):
        return False
    return True


def _valid(cell: tuple) -> bool:
    """A cell is in the universe exactly when no reviewed exclusion row matches it."""
    return not derivative_exclusions.is_excluded(_split_cell(cell))


_STEM_MAP = {
    ("cart_gto", "mol", "AE"): "H2_ae_ccpvdz_cart.h5",
    ("cart_gto", "mol", "ECP"): "H2_ecp_ccpvdz_cart.h5",
    ("sphe_gto", "mol", "AE"): "H2_ae_ccpvdz_sphe.h5",
    ("sphe_gto", "mol", "ECP"): "H2_ecp_ccpvtz.h5",
    ("sphe_pgto", "pbc_gamma", "AE"): "H2_pbc_ae_sphe.h5",
    ("sphe_pgto", "pbc_gamma", "ECP"): "lithium.h5",
    ("sphe_pgto_complex", "twist", "AE"): "H2_pbc_ae_k025_sphe.h5",
    ("sphe_pgto_complex", "twist", "ECP"): "lithium_k025.h5",
    ("cart_pgto", "pbc_gamma", "AE"): "H2_pbc_ae_cart.h5",
    ("cart_pgto", "pbc_gamma", "ECP"): "lithium_cart.h5",
    ("cart_pgto_complex", "twist", "AE"): "H2_pbc_ae_k025_cart.h5",
    ("cart_pgto_complex", "twist", "ECP"): "lithium_k025_cart.h5",
    ("composite", "pbc_gamma", "ECP"): _QE_STEM,
    # --- QE-projection leaves (plane-wave / B-spline / twist composite) built at test time
    # from the committed espresso outputs via _qe_proj_pieces (sentinel stems routed there by
    # _build_ham_for_cell). No h5, no QE run -- pure reader wiring of raw QE saves. AE vs ECP
    # is the Coulomb branch: ECP threads the parsed ccECP (Li), AE keeps the bare Coulomb
    # (H.coulomb-ae, sum(Z)==n_electron). Real-Gamma leaves come from an actual Gamma save
    # (Li_gamma); complex leaves from a genuine twist save (H-ae_k025 / Li_k025); the AE-Gamma
    # plane-wave / B-spline leaves from the committed all-electron Gamma save H-ae_gamma. ---
    ("plane_wave", "pbc_gamma", "ECP"): "QEPROJ:pw:Li_gamma:ECP",
    ("plane_wave", "pbc_gamma", "AE"): "QEPROJ:pw:H_gamma:AE",
    ("plane_wave_complex", "twist", "AE"): "QEPROJ:pw:H_k025:AE",
    ("plane_wave_complex", "twist", "ECP"): "QEPROJ:pw:Li_k025:ECP",
    ("bspline", "pbc_gamma", "ECP"): "QEPROJ:bspline:Li_gamma:ECP",
    ("bspline", "pbc_gamma", "AE"): "QEPROJ:bspline:H_gamma:AE",
    ("bspline_complex", "twist", "AE"): "QEPROJ:bspline:H_k025:AE",
    ("bspline_complex", "twist", "ECP"): "QEPROJ:bspline:Li_k025:ECP",
    ("composite", "twist", "AE"): "QEPROJ:pgto+pw:H_k025:AE",
    ("composite", "twist", "ECP"): "QEPROJ:pgto+pw:Li_k025:ECP",
}

# J3-orbital leaf kind per AO family (see _j3_orbital below): reuse the AO leaf for a
# molecular / real-pgto leaf, the complex leaf for twist, an MO-wrapped leaf for the
# composite and the plane-wave / B-spline leaves (those flat leaves do not expose a
# num_orb J3 accessor directly, so wrap them in an identity MOs_data).
_J3_KIND = {
    "cart_gto": "pgto",
    "sphe_gto": "pgto",
    "sphe_pgto": "pgto",
    "sphe_pgto_complex": "complex",
    "cart_pgto": "pgto",
    "cart_pgto_complex": "complex",
    "composite": "mo",
    "plane_wave": "mo",
    "plane_wave_complex": "mo",
    "bspline": "mo",
    "bspline_complex": "mo",
}

_QE_DIR = str(_ROOT / "tests" / "qe_example_files")
_LI_GAMMA_SAVE = os.path.join(_QE_DIR, "Li_gamma", "out", "li.save")
_LI_K025_SAVE = os.path.join(_QE_DIR, "Li_k025", "out", "li_k025.save")
_H_K025_SAVE = os.path.join(_QE_DIR, "H-ae_k025", "out", "h_k025.save")
_LI_GAMMA_UPF = os.path.join(_LI_GAMMA_SAVE, "Li.ccECP.upf")
_LI_K025_UPF = os.path.join(_LI_K025_SAVE, "Li.ccECP.upf")
_H_K025_UPF = os.path.join(_H_K025_SAVE, "H.coulomb-ae.UPF")
_H_GAMMA_SAVE = os.path.join(_QE_DIR, "H-ae_gamma", "out", "h_gamma.save")
_H_GAMMA_UPF = os.path.join(_H_GAMMA_SAVE, "H.coulomb-ae.UPF")

# sentinel stem -> (kind, save_dir, upf, ecp_flag, ik)
_QE_PROJ = {
    "QEPROJ:pw:Li_gamma:ECP": ("pw", _LI_GAMMA_SAVE, _LI_GAMMA_UPF, True, 1),
    "QEPROJ:pw:H_k025:AE": ("pw", _H_K025_SAVE, _H_K025_UPF, False, 1),
    "QEPROJ:pw:Li_k025:ECP": ("pw", _LI_K025_SAVE, _LI_K025_UPF, True, 1),
    "QEPROJ:pw:H_gamma:AE": ("pw", _H_GAMMA_SAVE, _H_GAMMA_UPF, False, 1),
    "QEPROJ:bspline:Li_gamma:ECP": ("bspline", _LI_GAMMA_SAVE, _LI_GAMMA_UPF, True, 1),
    "QEPROJ:bspline:H_k025:AE": ("bspline", _H_K025_SAVE, _H_K025_UPF, False, 1),
    "QEPROJ:bspline:Li_k025:ECP": ("bspline", _LI_K025_SAVE, _LI_K025_UPF, True, 1),
    "QEPROJ:bspline:H_gamma:AE": ("bspline", _H_GAMMA_SAVE, _H_GAMMA_UPF, False, 1),
    "QEPROJ:pgto+pw:H_k025:AE": ("pgto+pw", _H_K025_SAVE, _H_K025_UPF, False, 1),
    "QEPROJ:pgto+pw:Li_k025:ECP": ("pgto+pw", _LI_K025_SAVE, _LI_K025_UPF, True, 1),
    "QEPROJ:pgto+bspline:Li_gamma:ECP": ("pgto+bspline", _LI_GAMMA_SAVE, _LI_GAMMA_UPF, True, 1),
    "QEPROJ:pgto+bspline:Li_k025:ECP": ("pgto+bspline", _LI_K025_SAVE, _LI_K025_UPF, True, 1),
}


def _stem_available(stem: str) -> bool:
    """Whether all required source files exist now, independently of matrix selection."""
    if stem == _QE_STEM:
        return all(
            fixture_exists(os.path.join(_QE_SAVE, name)) for name in ("data-file-schema.xml", "wfc1.hdf5", "Li.ccECP.upf")
        )
    spec = _QE_PROJ.get(stem)
    if spec is None:
        return (_ROOT / "tests/trexio_example_files" / stem).is_file()
    _kind, save, upf, ecp, ik = spec
    ok = fixture_exists(os.path.join(save, "data-file-schema.xml")) and fixture_exists(os.path.join(save, f"wfc{ik}.hdf5"))
    if ecp or _kind.startswith("pgto+"):  # ECP parse and composite pGTO seeding both need the UPF
        ok = ok and fixture_exists(upf)
    return ok


def _resolve(cell: tuple, *, require_sources: bool = True):
    """Map a cell to ``(stem, with_jastrow, spec)`` for ``_build_ham``, or ``None`` if no
    committed builder path exists (fixture debt).

    NOTE on the Jastrow axis realizability: J1 and J2 are distinct universe tokens, but the
    only committed molecular builder (``_build_ham(with_jastrow=True)``) attaches J1 AND J2
    together (no isolated-channel source), so both tokens resolve to the same J1+J2 ham.
    This is a documented builder limitation, not a phantom cell -- the gate verdict is
    identical for both and the covering array still spans every (channel, *) pair.

    Returns ``(stem, with_jastrow, spec, gem)``; the ``gem`` token is threaded to
    ``_build_ham_for_cell`` which rewrites the base SD geminal into that representation. A
    dense-AO ``AGP`` cell whose AO family is not well-conditioned (``_agp_dense_constructible``)
    has no builder path (its a==b compare would be worthless).
    """
    ao, jas, ecp, bc, gem = _split_cell(cell)
    if gem == "AGP" and not _agp_dense_constructible(ao, bc):
        return None
    stem = _STEM_MAP.get((ao, bc, ecp))
    if stem is None:
        return None
    if require_sources and not _stem_available(stem):
        return None
    if jas == "none":
        return (stem, False, None, gem)
    if jas in ("J1", "J2"):
        return (stem, True, None, gem)  # committed builder couples J1+J2
    if jas == "J3poly":
        return (stem, True, {"j3": _J3_KIND[ao]}, gem)
    if jas == "DTN":
        return (stem, True, {"dtn": True}, gem)
    return None


def _buildable(cell) -> bool:
    return _resolve(cell) is not None


def _has_declared_builder(cell) -> bool:
    """Selection depends on reviewed builder wiring, never the worker's filesystem."""
    return _resolve(cell, require_sources=False) is not None


def _no_builder_reason(cell) -> str:
    """Precise reason a cell has no committed builder path (printed in the skip + inventory)."""
    ao, _jas, ecp, bc, gem = _split_cell(cell)
    if gem == "AGP" and not _agp_dense_constructible(ao, bc):
        return f"dense-AO-AGP not well-conditioned on AO family {ao!r} (ill-conditioned overlap => SVD-based a==b worthless)"
    return "no committed source for this (ao, bc, ecp) combination"


_PAIR_POSITIONS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))  # 4-axis SD core (ao, jas, ecp, bc)
_PAIR_POSITIONS5 = _PAIR_POSITIONS + ((0, 4), (1, 4), (2, 4), (3, 4))  # + the gem axis (position 4)


def _base_candidates() -> list[tuple[str, str, str, str]]:
    """The SORTED valid 4-tuple ``(ao, jas, ecp, bc)`` universe (no ``gem`` axis). The SD
    sub-matrix is generated over THESE tuples, so it is byte-identical to the 4-axis matrix.
    """
    aos, jas, ecps, bcs = _ao_families(), _jastrow_types(), list(_ECPS), list(_BCS)
    cands = [c for c in itertools.product(aos, jas, ecps, bcs) if _valid(c)]
    return sorted(cands)


def _candidates() -> list[tuple[str, str, str, str, str]]:
    """The SORTED valid 5-tuple ``(ao, jas, ecp, bc, gem)`` universe. ``SD`` / ``AGPn`` /
    ``SD`` / ``AGPn`` are admitted on every valid 4-tuple; ``AGP`` (dense-AO) only where
    ``_agp_dense_constructible``, and ``AGP_MO`` only where ``_agp_mo_constructible``.
    """
    cands = [c4 + (gem,) for c4 in _base_candidates() for gem in _GEMS if _valid(c4 + (gem,))]
    return sorted(cands)


def _required_pairs(cands, positions=_PAIR_POSITIONS) -> set:
    req = set()
    for c in cands:
        for i, j in positions:
            req.add((i, j, c[i], c[j]))
    return req


def _greedy_pairwise(cands, required, positions=_PAIR_POSITIONS) -> list[tuple]:
    """Deterministic greedy set-cover of the ``required`` pairs over ``positions``. ``required``
    is the STILL-UNCOVERED pair set (the caller may pre-subtract pairs a seed already covers), so
    seeding is just passing a reduced ``required`` -- selected cells only cover residual pairs.
    """
    uncovered = set(required)
    selected: list[tuple] = []
    while uncovered:
        best = None
        best_key = None
        for c in cands:  # cands is sorted -> tie-break is deterministic
            cov = sum(1 for i, j in positions if (i, j, c[i], c[j]) in uncovered)
            if cov == 0:
                continue
            key = (-cov, 0 if _has_declared_builder(c) else 1, c)
            if best_key is None or key < best_key:
                best_key, best = key, c
        selected.append(best)
        for i, j in positions:
            uncovered.discard((i, j, best[i], best[j]))
    return selected


def _three_way_twist(cands) -> list[tuple]:
    """Selective 3-way over ``{twist} x jas x {AE,ECP}``: one representative cell per
    (twist, jas, ecp) triple, preferring a buildable AO leaf, else the first sorted.
    """
    out: list[tuple] = []
    twist = [c for c in cands if c[3] == "twist"]
    for jas in _jastrow_types():
        for ecp in _ECPS:
            combo = [c for c in twist if c[1] == jas and c[2] == ecp]
            if not combo:
                continue
            combo_sorted = sorted(combo, key=lambda c: (0 if _has_declared_builder(c) else 1, c))
            out.append(combo_sorted[0])
    return out


# Real pGTO AO families that admit a periodic-Gamma pseudopotential fixture (lithium.h5 /
# lithium_cart.h5). The plain pairwise cover never builds a ``(pgto, J2|J3, ECP, pbc_gamma)``
# cell, so the periodic-ECP Jastrow c-axis (the site of the ``one_move_mesh.j3_mesh`` symmetric-fold
# bug -- a non-symmetric J3 block poisoned the mesh RATIO weight, throwing J1/J2/J3 ``dE_L/dc``
# off ``_auto``) went unasserted. This selective block forces it, permanently.
_PBC_ECP_PGTO_AOS = ("cart_pgto", "sphe_pgto")


def _selective_pbc_ecp_jastrow(cands) -> list[tuple]:
    """Selective ``{pbc_gamma} x {real pGTO AO} x jas x {ECP}`` block: one representative
    ``(pgto, jas, ECP, pbc_gamma)`` cell per (AO family, Jastrow channel), preferring a
    buildable stem. Mirrors ``_three_way_twist``'s discipline (additive on top of the pairwise
    core; the AGP/geminal ``gem`` axis is left at SD -- the c-axis fold under test is the
    Jastrow ratio, geminal-representation-independent). Asserts BOTH quantities (ln_Psi, e_L)
    and every axis for these cells, so the k=Gamma c-axis table of the generated coverage
    page stops showing ``-`` for the periodic-ECP pGTO J1/J2/J3/DTN e_L blocks.
    """
    out: list[tuple] = []
    pool = [c for c in cands if c[0] in _PBC_ECP_PGTO_AOS and c[2] == "ECP" and c[3] == "pbc_gamma"]
    for ao in _PBC_ECP_PGTO_AOS:
        for jas in _jastrow_types():
            if jas == "none":
                continue  # a Jastrow-free periodic-ECP cell is already covered (composite/pgto none)
            combo = [c for c in pool if c[0] == ao and c[1] == jas]
            if not combo:
                continue
            out.append(sorted(combo, key=lambda c: (0 if _has_declared_builder(c) else 1, c))[0])
    return out


def _is_heavy(cell) -> bool:
    """Heavy tail = every PERIODIC cell (``bc != mol``): periodic pGTO / composite / plane-wave /
    B-spline and all twist cells (incl. their AGPn / dense-AGP variants). MEASURED on this box
    (fp64, -n 8): a single periodic-pGTO cell's 8 (quantity, axis) traces cost ~190 s
    (compile-bound), and the QE composite is heavier still -- so the whole periodic/twist set blows
    the <=30 min matrix budget and is deferred to the heavy tier. The LIGHT pairwise core is the
    molecular set (cart/sphe GTO SD / AGPn / dense-AGP, ~155 s at -n 8): the molecular dense-AO-AGP
    carries a full dense lambda but the ham itself is tiny, so it stays light AND locally verifiable;
    the heavier composite dense-AGP is already periodic => heavy.
    """
    return _split_cell(cell)[3] != "mol"


def _generate_matrix() -> list[tuple[tuple, bool]]:
    """The generated cell list as ``[(cell, is_heavy), ...]``, DETERMINISTIC. Two phases:

    (1) SD core: the 4-axis pairwise cover + 3-way twist block, generated over the
        4-tuple universe, then tagged ``gem == SD`` (byte-identical tuples/order).
    (2) gem extension: fold ``gem`` into a 5-axis pairwise cover SEEDED by the SD core -- subtract
        the pairs the SD core already covers, then greedily add non-SD (AGPn / AGP / AGP_MO) cells for the
        residual position-4 pairs. Additive: SD sub-matrix untouched, no cartesian blow-up.
    """
    # (1) SD core over the 4-tuple universe (unchanged), then tag gem=SD.
    base = _base_candidates()
    pairwise = _greedy_pairwise(base, _required_pairs(base, _PAIR_POSITIONS), _PAIR_POSITIONS)
    three = _three_way_twist(base)
    sel_pbc = _selective_pbc_ecp_jastrow(base)  # forced periodic-ECP pGTO Jastrow c-axis coverage
    ordered: list[tuple] = []
    seen = set()
    for c in pairwise + three + sel_pbc:  # pairwise core, then 3-way twist, then pbc-ECP block (dedup)
        if c not in seen:
            seen.add(c)
            ordered.append(c + ("SD",))
    # (2) gem extension: seed the 5-axis cover with the SD core, add cells for residual gem pairs.
    cands5 = _candidates()
    uncovered = _required_pairs(cands5, _PAIR_POSITIONS5)
    for c in ordered:
        for i, j in _PAIR_POSITIONS5:
            uncovered.discard((i, j, c[i], c[j]))
    seen5 = set(ordered)
    for c in _greedy_pairwise(cands5, uncovered, _PAIR_POSITIONS5):
        if c not in seen5:
            seen5.add(c)
            ordered.append(c)
    return [(c, _is_heavy(c)) for c in ordered]


def _cell_id(cell) -> str:
    return "__".join(cell)


def _matrix_hash(matrix) -> str:
    return hashlib.sha256(repr([tuple(c) for c, _ in matrix]).encode()).hexdigest()

"""Shared base for the machine-generated route-fidelity fixture matrix.

This module holds EVERYTHING the matrix generation + per-cell route-fidelity test needs, so the
per-system split files (``test_analytic_grad_matches_autograd_generated_{molecular,periodic,twist}.py``)
each import from here, filter ``_generate_matrix()`` to their own boundary condition, and parametrize
the ``run_matrix_cell`` body over only that subset. It is a leading-underscore module -- NOT collected
by pytest; it defines no ``test_*`` functions.

Where ``test_analytic_grad_matches_autograd_curated.py`` is a HAND-PICKED set of 15 fixtures, the
GENERATED matrix enumerates the ENTIRE channel universe by introspecting the codebase, thins it with a
DETERMINISTIC pairwise covering array, and -- for every generated cell -- OBSERVES the production
analytic route directly against its ``_auto`` oracle, with NO capability-gate pre-check:

  * the analytic path is called directly and must equal the ``_auto`` twin (a == b,
    switch-ON == switch-OFF) at the zone-derived tol -- a HARD assertion, not an xfail.
  * a ``NotImplementedError`` raised by the analytic path is a "RAISE cell" -- the
    MECHANICAL mission-remainder inventory (recorded, then re-raised as a hard failure;
    mission complete when the raise-cell count is 0).

This detects BOTH classes of gap as RED, not xfail: an analytic path that RAISES is a
RAISE cell (mission remainder, re-raised); an analytic path that RUNS but DIVERGES from
``_auto`` (the twist x DTN class of bug) is a FINDING (recorded, then re-raised as a hard
``AssertionError`` failure). There is no capability registry -- analytic coverage is this
behavioral matrix (plus the raise-vs-no-raise ``test_no_autodiff_in_production``).

Universe axes (INTROSPECTED, not hand-enumerated -- ``_ao_families`` / ``_jastrow_types``):
  * ``ao``  -- every concrete ``AOs_*_data`` leaf family in ``jqmc.atomic_orbital``
              (cart/sphe GTO, real & complex pGTO, composite, plane-wave, B-spline).
  * ``jas`` -- every ``jastrow_*_data`` channel field of ``jqmc.jastrow_factor.Jastrow_data``
              (J1, J2, J3poly, DTN) plus the ``none`` baseline.
  * ``ecp`` -- ``{AE, ECP}``.
  * ``bc``  -- ``{mol, pbc_gamma, twist}`` (all-electron/pseudo x boundary condition).
  * ``gem`` -- the geminal REPRESENTATION ``{SD, AGPn, AGP, AGP_MO}``. ``SD`` is the plain
              single determinant the committed source produces; ``AGPn`` is the diagonal
              natural-orbital form (``AGPn_data``, ``lambda_diag``); ``AGP`` is ALWAYS the
              DENSE-AO-AGP -- a full dense pairing ``lambda_{ab}`` in the non-orthogonal AO
              basis (``Geminal_data.convert_from_MOs_to_AOs`` + the 8-set full-rank lift).
              ``AGP_MO`` is the same dense ``AGP_data.lambda_matrix`` variational family
              represented over explicit ``MOs_data`` leaves.

The geminal REPRESENTATION is a real ``gem`` axis (above). ``SD`` and ``AGPn`` are broadly valid (any fixture); ``AGP``
(dense-AO) is valid ONLY where a WELL-CONDITIONED dense AO-AGP is constructible; ``AGP_MO``
is valid everywhere the base SD MO fixture is constructible. The
dense-lambda natural-orbital SVD both the analytic and ``_auto`` derivatives differentiate
is garbage on a near-singular AO overlap (the image-truncated periodic bare-pGTO basis),
so ``AGP`` is admitted only on the well-conditioned molecular GTO leaves and the QE
pgto(+)PW composite (the exact 8-set dense-AGP basis); see ``_agp_dense_constructible``.
MPC is a ctx-driven correction (``mpc_ctx``), NOT a ham channel, so it is OUT OF SCOPE for
the matrix (its oracle is FD-arbitrated, not the ``_auto`` twin; see the 8-set harness).

Physical-constraint predicate (``_bc_allowed`` -- MIRRORS what the capability gate itself
assumes): a complex-Bloch leaf implies ``twist`` (k != 0); a real periodic leaf
(pGTO / plane-wave / B-spline) implies a periodic boundary; a molecular GTO leaf implies
``mol``; a composite (pGTO(+)PW) may be either ``pbc_gamma`` or ``twist``. Cells failing
the predicate are dropped from the universe (they are not physically constructible).

Covering array (``_generate_matrix``): a DETERMINISTIC greedy pairwise (2-way) cover over
the SORTED universe -- no randomness, no committed table, so it is reproducible AND
auto-expands when a new channel is added to the code. Selection key is
``(-pairs_covered, not buildable, sorted_tuple)``: maximize newly-covered pairs, then
PREFER a cell we can actually build (so the real a==b assertions land on concrete hams
rather than "no builder" skips), then lexicographic for a total order. On top of the
pairwise core we add the SELECTIVE 3-way ``{twist} x jas x {AE,ECP}`` block (the twist
dtype propagates globally, so its interaction with every Jastrow x AE/ECP triple is
worth explicit coverage). Twist cells are the HEAVY tail (dtype-doubled traces); the
non-twist pairwise core is ``light``.

The ``gem`` axis is folded into the SAME pairwise cover WITHOUT perturbing the
SD sub-matrix: the 4-axis ``(ao, jas, ecp, bc)`` pairwise core + 3-way twist
block are generated over the 4-tuple universe and tagged ``gem == SD`` (byte-identical tuples and
order), then that SD core SEEDS a 5-axis pairwise cover -- only the still-uncovered
``gem``-involving pairs (position 4 x AGPn / AGP / AGP_MO) pull in ADDITIONAL non-SD cells. This is
purely additive: representative ``gem`` coverage with no cartesian blow-up and the SD cells
untouched.

Builder: each cell is mapped to a committed builder path via ``_build_ham`` imported from
``_grad_fidelity_harness`` (the shared harness module). A cell with no committed builder path is SKIPPED with reason
"no builder" and COUNTED (future fixture debt, not a silent hole).

Memory discipline: the bridge inner-jit LRU + XLA programs are UNBOUNDED; walking dozens
of distinct hams in one process exhausts LLVM mmap (measured). ``bridge.cache_clear()``
runs per-cell teardown; combined with xdist workers this holds.
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
import functools
import hashlib
import itertools
import os

import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine import (
    CaseReceipt,
    ComparisonBinding,
    ElectronConfigurationBinding,
    InputBinding,
    ProductionRouteCapture,
    observe_allclose,
    observe_electron_configurations,
)
from gdd.engine.fixture_observations import validate_matrix_fixture_route
import jqmc.atomic_orbital as _atomic_orbital
from jqmc._precision import get_tolerance_min
from jqmc.derivatives import bridge
from jqmc.determinant import AGP_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel._gdd_adapters import (
    current_precision_mode,
    guarantee_id_for_manifest_family,
)

# Builder + comparison core -- IMPORTED from the shared harness (single source of the ham
# builders and the route-faithful a==b comparator).
from tests.test_jqmc_kernel._grad_fidelity_harness import (
    _NW,
    _QE_AVAILABLE,
    _QE_STEM,
    _assert_axis_allclose,
    _batched_geminal_inv,
    _build_ham,
    _draw_batch,
    _flags_for_axis,
    _make_full_rank_ao_agp,
)

# Cell enumeration, stems, and buildability now live in the GDD engine (requirement
# side).  This module re-exports them for the split matrix test modules.
from gdd.engine.derivative_matrix import (  # noqa: F401  (re-exported enumeration API)
    _AGP_MO_AO_FAMILIES,
    _BCS,
    _ECPS,
    _GEMS,
    _J3_KIND,
    _QE_DIR,
    _QE_PROJ,
    _STEM_MAP,
    _agp_dense_constructible,
    _ao_families,
    _bc_allowed,
    _buildable,
    _cell_id,
    _generate_matrix,
    _jastrow_types,
    _matrix_hash,
    _no_builder_reason,
    _resolve,
    _split_cell,
    _valid,
)
from tests._ecp_core_electrons import core_electrons_of
from tests._test_data_cache import qe_example_files_root

_QUANTITIES = ("ln_Psi", "e_L")
_AXES = ("c", "r", "R", "eps")
# Geminal-representation axis: SD = plain single determinant (the committed default);
# AGPn = diagonal natural-orbital form (``AGPn_data``); AGP = DENSE-AO-AGP (full dense lambda
# in the non-orthogonal AO basis); AGP_MO = dense ``AGP_data.lambda_matrix`` over ``MOs_data``.
# ``AGP`` stays the AO spelling so node IDs are stable; the MO form is the separate
# ``AGP_MO`` token.

# Reviewed route name for the wrapped analytical production call; the electron
# binding declares it, so CaseReceipt.validate demands the capture receipt.
_MATRIX_PRODUCTION_ROUTE = "derivatives.matrix.analytical-production-route.v1"

_COMPARISON_ZONES = (
    "ao_eval",
    "jastrow_eval",
    "jastrow_grad_lap",
    "det_eval",
    "coulomb",
    "wf_kinetic",
)


def _expanded_mo_rows(mos: MOs_data, rank: int, *, seed: int) -> MOs_data:
    """Return ``rank`` deterministic, linearly independent MO rows for a test fixture."""
    coefficients = np.asarray(mos.mo_coefficients)
    if coefficients.shape[0] >= rank:
        expanded = coefficients[:rank]
    else:
        rng = np.random.default_rng(seed)
        extra_shape = (rank - coefficients.shape[0], coefficients.shape[1])
        extra = rng.normal(size=extra_shape)
        if np.iscomplexobj(coefficients):
            extra = extra + 1j * rng.normal(size=extra_shape)
        seeds = np.vstack([coefficients, extra]).astype(coefficients.dtype, copy=False)
        # Euclidean QR is sufficient for this deterministic derivative fixture: it prevents
        # invented virtual rows from being linearly dependent without assuming a particular
        # AO-overlap implementation for every GTO/PW/B-spline/composite leaf.
        expanded = np.linalg.qr(seeds.T, mode="reduced")[0].T
    return MOs_data(
        num_mo=rank,
        aos_data=mos.aos_data,
        mo_coefficients=np.ascontiguousarray(expanded),
    )


def _make_dense_mo_agp(sd, ion_positions=None) -> AGP_data:
    """Construct a genuine dense AGP matrix over explicit MO leaves.

    Whenever both AO spaces permit it, use at least two MOs and a non-diagonal paired
    block.  This avoids reducing the MO-AGP fixture to an AGPn-equivalent diagonal or a
    one-by-one type witness.  One-dimensional source spaces remain valid rank-one AGPs.

    When ``ion_positions`` is given, the dense block is WHITENED at the ion-centred
    reference configuration: ``lambda = W T W^H`` (time-reversal) or
    ``W_u T pinv(M_dn0)`` (independent spins), where ``W = pinv(M_up0^T)`` and ``M0``
    are the MO values at the reference points, so that ``G(reference) ~ T``. The
    electron-configuration screening (``cond_2(G) < 700``) draws ion-centred jittered
    configurations; without whitening ``cond(G) ~ cond(M)^2``, and the near-collinear
    plane-wave / B-spline all-electron bands (cond(M) ~ 300-400) leave NO admissible
    configuration in the 300-candidate budget -- the AGP_MO cells on those leaves were
    unconstructible from the day they were enumerated. ``T`` keeps the same dense
    couplings as before, so the fixture stays a genuinely non-diagonal MO-AGP.
    """
    num_up = int(sd.num_electron_up)
    num_dn = int(sd.num_electron_dn)
    max_rank = min(
        int(sd.orb_data_up_spin.aos_data.num_ao),
        int(sd.orb_data_dn_spin.aos_data.num_ao),
    )
    rank = min(max(max(num_up, num_dn), 2), max_rank)
    mos_up = _expanded_mo_rows(sd.orb_data_up_spin, rank, seed=7103)
    if (
        bool(sd.time_reversal)
        and np.asarray(mos_up.mo_coefficients).shape[1] == np.asarray(sd.orb_data_dn_spin.mo_coefficients).shape[1]
    ):
        mos_dn = MOs_data(
            num_mo=rank,
            aos_data=sd.orb_data_dn_spin.aos_data,
            mo_coefficients=np.ascontiguousarray(np.conj(mos_up.mo_coefficients)),
        )
    else:
        mos_dn = _expanded_mo_rows(sd.orb_data_dn_spin, rank, seed=7109)

    complex_fixture = np.iscomplexobj(mos_up.mo_coefficients) or np.iscomplexobj(mos_dn.mo_coefficients)
    dtype = np.complex128 if complex_fixture else np.float64

    def _dense_target(rows: int, columns: int) -> np.ndarray:
        target = np.zeros((rows, columns), dtype=dtype)
        square = min(rows, columns)
        target[:square, :square] = np.diag(np.linspace(1.25, 0.55, square))
        for row in range(square):
            for column in range(row + 1, square):
                value = 0.17 + (0.08j if complex_fixture else 0.0)
                target[row, column] = value
                target[column, row] = np.conj(value)
        return target

    unpaired_count = num_up - num_dn
    if ion_positions is None:
        paired = _dense_target(rank, rank)
        unpaired = np.zeros((rank, unpaired_count), dtype=dtype)
        for column in range(unpaired_count):
            unpaired[min(num_dn + column, rank - 1), column] = 1.0
    else:
        from jqmc.molecular_orbital import compute_MOs

        positions = np.asarray(ion_positions, dtype=np.float64)
        reference_up = positions[np.arange(num_up) % len(positions)]
        reference_dn = positions[np.arange(num_dn) % len(positions)]
        m_up0 = np.asarray(compute_MOs(mos_up, reference_up))  # (rank, num_up)
        w_up = np.linalg.pinv(m_up0.T)  # (rank, num_up)
        if bool(sd.time_reversal) and num_up == num_dn:
            paired = (w_up @ _dense_target(num_up, num_dn) @ np.conj(w_up).T).astype(dtype)
        else:
            m_dn0 = np.asarray(compute_MOs(mos_dn, reference_dn))  # (rank, num_dn)
            paired = (w_up @ _dense_target(num_up, num_dn) @ np.linalg.pinv(m_dn0)).astype(dtype)
        unpaired_target = np.zeros((num_up, unpaired_count), dtype=dtype)
        for column in range(unpaired_count):
            unpaired_target[num_dn + column, column] = 1.0
        unpaired = (w_up @ unpaired_target).astype(dtype)
    return AGP_data.build(
        num_electron_up=num_up,
        num_electron_dn=num_dn,
        orb_data_up_spin=mos_up,
        orb_data_dn_spin=mos_dn,
        lambda_matrix=np.hstack([paired, unpaired]),
        time_reversal=bool(sd.time_reversal),
    )


@dataclasses.dataclass(frozen=True)
class MatrixCaseBinding:
    """Fully resolved GDD metadata for one generated pytest parameter case."""

    guarantee_id: str
    obligation_id: str
    pytest_node_id: str
    input_binding: InputBinding
    electron_configuration: ElectronConfigurationBinding
    comparison: ComparisonBinding


# --------------------------------------------------------------------------- #
# Builder mapping: (ao, bc, ecp) -> committed trexio/QE stem. Only the intersections
# with a committed tiny source are buildable; every other cell is "no builder" debt.
# --------------------------------------------------------------------------- #

# Adapter from the builder stems to the committed source IDs governed by
# gdd/ssot/fixtures/sources.yaml. The static GDD checker verifies every value against
# that registry; a new stem therefore cannot enter the matrix without an explicit
# source binding.
_GDD_SOURCE_ID_BY_STEM = {
    "H2_ae_ccpvdz_cart.h5": "WF-SRC-H2-AE-CART-MOL",
    "H2_ecp_ccpvdz_cart.h5": "WF-SRC-H2-ECP-CART-MOL",
    "H2_ae_ccpvdz_sphe.h5": "WF-SRC-H2-AE-SPHE-MOL",
    "H2_ecp_ccpvtz.h5": "WF-SRC-H2-ECP-SPHE-MOL",
    "H2_pbc_ae_sphe.h5": "WF-SRC-H2-AE-SPHE-GAMMA",
    "lithium.h5": "WF-SRC-LI-ECP-SPHE-GAMMA",
    "H2_pbc_ae_k025_sphe.h5": "WF-SRC-H2-AE-SPHE-NON-GAMMA",
    "lithium_k025.h5": "WF-SRC-LI-ECP-SPHE-NON-GAMMA",
    "H2_pbc_ae_cart.h5": "WF-SRC-H2-AE-CART-GAMMA",
    "lithium_cart.h5": "WF-SRC-LI-ECP-CART-GAMMA",
    "H2_pbc_ae_k025_cart.h5": "WF-SRC-H2-AE-CART-NON-GAMMA",
    "lithium_k025_cart.h5": "WF-SRC-LI-ECP-CART-NON-GAMMA",
    _QE_STEM: "WF-SRC-LI-ECP-GAMMA",
    "QEPROJ:pw:Li_gamma:ECP": "WF-SRC-LI-ECP-GAMMA",
    "QEPROJ:pw:H_gamma:AE": "WF-SRC-H-AE-GAMMA",
    "QEPROJ:pw:H_k025:AE": "WF-SRC-H-AE-NON-GAMMA",
    "QEPROJ:pw:Li_k025:ECP": "WF-SRC-LI-ECP-NON-GAMMA",
    "QEPROJ:bspline:Li_gamma:ECP": "WF-SRC-LI-ECP-GAMMA",
    "QEPROJ:bspline:H_gamma:AE": "WF-SRC-H-AE-GAMMA",
    "QEPROJ:bspline:H_k025:AE": "WF-SRC-H-AE-NON-GAMMA",
    "QEPROJ:bspline:Li_k025:ECP": "WF-SRC-LI-ECP-NON-GAMMA",
    "QEPROJ:pgto+pw:H_k025:AE": "WF-SRC-H-AE-NON-GAMMA",
    "QEPROJ:pgto+pw:Li_k025:ECP": "WF-SRC-LI-ECP-NON-GAMMA",
    "QEPROJ:pgto+bspline:Li_gamma:ECP": "WF-SRC-LI-ECP-GAMMA",
    "QEPROJ:pgto+bspline:Li_k025:ECP": "WF-SRC-LI-ECP-NON-GAMMA",
}

# --------------------------------------------------------------------------- #
# QE-projection leaf builders (plane-wave / B-spline / twist composite). Committed espresso
# outputs under qe_example_files are loaded, projected onto the requested basis kind, and
# wrapped into a Hamiltonian_data whose Coulomb branch is AE (bare) or ECP (parsed ccECP).
# The 8-set harness's ``_build_ham`` (imported, never edited) has no plane-wave / B-spline
# path, so these leaves are assembled LOCALLY here (the matrix module's own builder).
# --------------------------------------------------------------------------- #

# Basis sizing kept tiny -- route fidelity (analytic == _auto) is independent of basis size,
# so the plane-wave / composite-PW blocks are truncated to the lowest-|G| columns.
_QE_PW_CAP = 48  # lowest-|G| plane waves retained for a pure plane-wave leaf
_QE_CMP_PW_CAP = 40  # PW columns retained in a pGTO(+)PW composite leaf
_QE_PGTO_ALPHA_WINDOW = (1.0, 8.0)  # pGTO primitives kept resolvable AND PW-independent (composite)


def _gcut_for_count(ref, n_target: int) -> float | None:
    """``g_cut`` (a ``|G|`` norm) keeping approximately the ``n_target`` lowest-|G| plane
    waves of ``ref``; ``None`` if the wfc already has fewer G-vectors (keep them all).
    """
    gnorm = np.linalg.norm(ref.wfc.g_vectors(ref.cell), axis=1)
    if gnorm.size <= n_target:
        return None
    return float(np.partition(gnorm, n_target)[n_target])


def _window_shells(shells_per_atom, lo: float, hi: float):
    """Keep only the pGTO primitives with ``lo <= exponent <= hi`` (composite conditioning,
    mirroring ``_grad_fidelity_harness._window_shells``); shells left empty are dropped.
    """
    out = []
    for shells in shells_per_atom:
        kept = []
        for s in shells:
            pairs = [(e, c) for e, c in zip(s.exponents, s.coefficients) if lo <= e <= hi]
            if pairs:
                kept.append(
                    dataclasses.replace(s, exponents=tuple(p[0] for p in pairs), coefficients=tuple(p[1] for p in pairs))
                )
        out.append(kept)
    return out


@functools.cache
def _qe_proj_pieces(stem: str):
    """``(structure, aos, geminal, coulomb_potential)`` for a QE-projection sentinel, built
    ONCE per process. Mirrors the light ``test_qe_to_jqmc`` / ``test_integration_qe_periodic``
    assembly: load the espresso save, project the occupied bands onto the requested basis
    ``kind``, and wrap with the AE (bare Coulomb) or ECP (parsed ccECP) potential + Ewald.
    """
    from jqmc.pw_pgto_projection import project_pw_to_mixed, project_pw_to_pgto_spline, project_pw_to_pw, project_pw_to_spline
    from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit, parse_gaussinfo

    kind, save, upf, ecp, ik = _QE_PROJ[stem]
    # _QE_PROJ names the checkout tree; read it through the session mirror that holds the pseudopotentials.
    save, upf = (os.path.join(qe_example_files_root(), os.path.relpath(path, _QE_DIR)) for path in (save, upf))
    ref = load_qe_reference(save, ik=ik)
    if kind == "pw":
        proj = project_pw_to_pw(ref, g_cut=_gcut_for_count(ref, _QE_PW_CAP))
    elif kind == "bspline":
        proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
    elif kind == "pgto+pw":
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        strat = UPFExponentInit({ref.atom_labels[0].lower(): upf}, g_target=g_max)
        shells = _window_shells([strat.shells_for_element(lbl) for lbl in ref.atom_labels], *_QE_PGTO_ALPHA_WINDOW)
        proj = project_pw_to_mixed(ref, shells, g_cut=_gcut_for_count(ref, _QE_CMP_PW_CAP), drop_diffuse=False)
    elif kind == "pgto+bspline":
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        strat = UPFExponentInit({ref.atom_labels[0].lower(): upf}, g_target=g_max)
        shells = _window_shells([strat.shells_for_element(lbl) for lbl in ref.atom_labels], *_QE_PGTO_ALPHA_WINDOW)
        proj = project_pw_to_pgto_spline(ref, shells, mesh_factor=0.5, order=5)
    else:
        raise ValueError(f"unsupported QE projection kind {kind!r}")
    structure_data, aos_data, geminal_data = build_wavefunction_from_projection(ref, proj, kind=kind)
    ecp_by_element = {ref.atom_labels[0]: parse_gaussinfo(upf)} if ecp else None
    base_ham = build_hamiltonian_from_wavefunction(
        structure_data, aos_data, geminal_data, j1_param=None, j2_param=None, ecp_by_element=ecp_by_element
    )
    return structure_data, aos_data, geminal_data, base_ham.coulomb_potential_data


def _j3_orbital(aos_data, kind: str):
    """The J3 orbital leaf: the AO leaf itself (``pgto`` / ``complex``) or that leaf wrapped
    in an identity ``MOs_data`` (``mo``). Mirrors ``_grad_fidelity_harness._j3_orbital``.
    """
    if kind == "mo":
        dtype = np.complex128 if _atomic_orbital.is_complex_aos(aos_data) else np.float64
        return MOs_data.build(aos_data=aos_data, mo_coefficients=np.eye(aos_data.num_ao, dtype=dtype))
    return aos_data


def _make_j3(orb_data) -> Jastrow_three_body_data:
    """An ACTIVE three-body Jastrow on ``orb_data`` (mirrors ``_grad_fidelity_harness._make_j3``)."""
    rng = np.random.default_rng(5)
    n = int(orb_data._num_orb)
    jm = rng.normal(size=(n, n + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    from jqmc.jastrow_factor import _phase_free_j3_orbitals

    return Jastrow_three_body_data(orb_data=_phase_free_j3_orbitals(orb_data), j_matrix=jm)


_BSPLINE_J3_AMPLITUDE_TRANSFORMATION = "derivatives.matrix.restore-bspline-j3-cell-normalized-amplitude.v1"


def _restores_bspline_j3_amplitude(stem: str, spec: dict | None) -> bool:
    """Whether a matrix cell carries an active J3 on a QE-projected B-spline leaf."""
    return stem.startswith("QEPROJ:bspline:") and bool((spec or {}).get("j3"))


def _restore_bspline_j3_amplitude(ham: Hamiltonian_data) -> Hamiltonian_data:
    """Restore the matrix J3 function on a QE B-spline leaf to its sized amplitude.

    The fixture J3 coupling of ``_make_j3`` is a seed-5 draw of O(0.1) that was
    sized for the raw QE band amplitude. The B-spline
    AO leaf carries the unit-cell normalization Omega^(-1/2), so on that leaf the
    J3 pair block shrank by Omega and the one-body column by Omega^(1/2). In
    mixed precision the J3 share of the e_L nuclear (R) gradient then fell below
    the medium comparison tolerance (0.66 and 0.14 of it for the AE Gamma and
    AE twist cells), so an analytic e_L/R route that drops J3 matched the
    autodiff twin. Rescaling the pair block by Omega and the one-body
    column by Omega^(1/2) restores exactly the J3 function of the un-normalized
    leaf. The plane-wave and pGTO leaves carry no such normalization and are untouched.
    """
    from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_bspline_data

    wf = ham.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    leaf = j3.orb_data.aos_data if isinstance(j3.orb_data, MOs_data) else j3.orb_data
    assert isinstance(leaf, (AOs_bspline_data, AOs_bspline_complex_data)), type(leaf).__name__
    omega = abs(float(np.linalg.det(np.asarray(ham.structure_data.cell))))
    jm = np.array(j3.j_matrix)
    jm[:, :-1] *= omega
    jm[:, -1] *= np.sqrt(omega)
    jastrow = wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm))
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=jastrow))
    ham.sanity_check()
    return ham


def _make_dtn(structure_data, coulomb_potential_data) -> Jastrow_DTN_data:
    """An ACTIVE DTN polynomial Jastrow (u/chi/f; mirrors ``_grad_fidelity_harness._make_dtn``)."""
    atomic_numbers = np.asarray(structure_data.atomic_numbers)
    unique_atomic_numbers = np.unique(atomic_numbers)
    effective_charges = np.asarray(coulomb_potential_data._effective_charges)
    return Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=tuple(float(np.mean(effective_charges[atomic_numbers == Z])) for Z in unique_atomic_numbers),
        pseudized_per_species=tuple(
            bool(np.all(coulomb_potential_data.pseudized_per_atom[atomic_numbers == Z])) for Z in unique_atomic_numbers
        ),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=2,
        n_order_chi=2,
        n_order_eN=2,
        n_order_ee=2,
        L_u=None,
        L_chi=None,
        L_f=None,
        random_init=True,
        seed=13,
    )


def _apply_gem(ham: Hamiltonian_data, gem: str) -> Hamiltonian_data:
    """Rewrite the ham's geminal into the requested representation (``gem`` axis). The base
    ham is always built with the committed ``SD`` geminal (MO-rep, ``MOs_data`` leaves); this
    swaps in the diagonal natural-orbital ``AGPn`` (``determinant.to_AGPn``), a dense
    ``AGP_data`` over the same explicit MOs (``AGP_MO``), or the well-conditioned dense-AO-AGP
    (the 8-set ``_make_full_rank_ao_agp`` -- ``convert_from_MOs_to_AOs`` + the full-rank lift,
    IMPORTED not duplicated). ``SD`` is a no-op. Same wavefunction otherwise.
    """
    if gem == "SD":
        return ham
    base_gem = ham.wavefunction_data.geminal_data
    if gem == "AGPn":
        # An open-shell occupied-only SD deliberately has no minority-spin
        # virtual rows, so it cannot be promoted exactly by production
        # ``to_AGPn``.  This matrix needs a genuine common-rank AGPn fixture;
        # construct its extra test orbitals explicitly instead of reintroducing
        # the removed persistent SD virtual tail.
        from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd

        new_gem = build_explicit_common_rank_agpn_from_sd(base_gem)
    elif gem == "AGP_MO":
        new_gem = _make_dense_mo_agp(base_gem, ion_positions=ham.structure_data.positions)
        assert isinstance(new_gem.orb_data_up_spin, MOs_data)
        assert isinstance(new_gem.orb_data_dn_spin, MOs_data)
    elif gem == "AGP":
        new_gem = _make_full_rank_ao_agp(base_gem)
    else:
        raise ValueError(f"unknown geminal representation token: {gem}")
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=new_gem))


def _build_ham_for_cell(stem: str, with_jastrow: bool, spec: dict | None, gem: str = "SD") -> Hamiltonian_data:
    """Build the cell ham. Trexio / QE-composite stems delegate to the imported 8-set
    ``_build_ham`` (never edited); QE-projection plane-wave / B-spline / twist-composite
    sentinels are assembled locally from ``_qe_proj_pieces`` with the SAME Jastrow recipe the
    8-set builder uses (J1+J2 coupled, optional active J3 / DTN). The base SD geminal is then
    rewritten into the requested ``gem`` representation via ``_apply_gem``.
    """
    if stem not in _QE_PROJ:
        return _apply_gem(_build_ham(stem, with_jastrow, spec), gem)
    spec = spec or {}
    structure_data, aos_data, geminal_data, coulomb_potential_data = _qe_proj_pieces(stem)
    j3 = _make_j3(_j3_orbital(aos_data, spec["j3"])) if spec.get("j3") else None
    dtn = _make_dtn(structure_data, coulomb_potential_data) if spec.get("dtn") else None
    if with_jastrow:
        n_atom = len(structure_data.positions)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
        )
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
            jastrow_1b_type="exp",
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=dtn
        )
    else:
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=j3, jastrow_dtn_data=dtn
        )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)
    return _apply_gem(ham, gem)


# Currently EMPTY. A tolerance level that passes on one environment and fails on another is
# calibration, not a proof, so a cell whose agreement is limited by the conditioning of its
# fixture (e.g. cond_2(lambda) ~ 5e4 of a whitened AGP_MO fixture) is left out of the matrix
# rather than listed here. The mechanism serves a case that genuinely needs a reviewed
# numerical precondition.
CONDITIONED_CELLS: dict[tuple[str, str, str], dict] = {}


def matrix_obligation_id(cell, quantity: str, axis: str) -> str:
    """The matrix obligation id of one cell x (quantity, axis); shared with the derivative registry."""
    family = {"ln_Psi": "ln_Psi", "e_L": "e_L"}[quantity]
    return f"{family}|matrix|{_cell_id(tuple(cell))}|axis={axis}"


def matrix_case_binding(cell, quantity: str, axis: str) -> MatrixCaseBinding:
    """Resolve the GDD bindings consumed by one generated matrix pytest case."""
    cell = tuple(cell)
    resolved = _resolve(cell)
    if resolved is None:
        raise AssertionError(f"GDD matrix case has no input builder: {_cell_id(cell)}")
    stem, with_jastrow, spec, gem = resolved
    source_id = _GDD_SOURCE_ID_BY_STEM.get(stem)
    if source_id is None:
        raise AssertionError(f"GDD matrix stem has no committed source ID: {stem}")
    ao, jas, _ecp, bc, _gem = _split_cell(cell)
    family = {"ln_Psi": "ln_Psi", "e_L": "e_L"}[quantity]
    guarantee_id = guarantee_id_for_manifest_family(family)
    obligation_id = matrix_obligation_id(cell, quantity, axis)
    filename = {
        "mol": "test_analytic_grad_matches_autograd_generated_molecular.py",
        "pbc_gamma": "test_analytic_grad_matches_autograd_generated_periodic.py",
        "twist": "test_analytic_grad_matches_autograd_generated_twist.py",
    }[bc]
    pytest_node_id = f"tests/test_jqmc_kernel/{filename}::test_matrix_cell[{_cell_id(cell)}-{quantity}-{axis}]"

    transformations = []
    if stem.startswith("QEPROJ:") or stem == _QE_STEM:
        transformations.append("derivatives.matrix.project-committed-qe-wavefunction.v1")
    if gem != "SD":
        representation = "agp-mo" if gem == "AGP_MO" else gem.lower()
        transformations.append(f"derivatives.matrix.convert-determinant-to-{representation}.v1")
    if with_jastrow:
        # The builder activates J1+J2 for J1/J2 rows and adds the selected
        # J3/DTN component for those rows. This records the actual transformation.
        transformations.append(f"derivatives.matrix.attach-jastrow-{jas.lower()}.v1")
    if spec:
        transformations.append("derivatives.matrix.apply-reviewed-builder-spec.v1")
    if _restores_bspline_j3_amplitude(stem, spec):
        transformations.append(_BSPLINE_J3_AMPLITUDE_TRANSFORMATION)

    input_binding = InputBinding(
        binding_id=f"DERIV-MATRIX-INPUT::{_cell_id(cell)}",
        kind="committed-artifact",
        source_ids=(source_id,),
        artifact_format="qe-save" if stem.startswith("QEPROJ:") or stem == _QE_STEM else "trexio",
        builder_operation="tests.test_jqmc_kernel._generated_matrix_base._build_ham_for_cell",
        transformations=tuple(transformations),
    )
    electron_configuration = ElectronConfigurationBinding(
        binding_id="DERIV-MATRIX-ELECTRONS::seed=0::count=2",
        mode="required",
        generation_pattern="direct",
        position_source_kind="seeded-random",
        generator_operation="tests.test_jqmc_kernel._grad_fidelity_harness._draw_batch",
        generator_arguments=(("distribution", "normal"), ("scale", 0.7)),
        random_algorithm="numpy.random.PCG64",
        seed=0,
        configuration_count=_NW,
        geminal_conditioning_mode="not-limited",
        production_route_capture=_MATRIX_PRODUCTION_ROUTE,
    )
    _conditioned = CONDITIONED_CELLS.get((_cell_id(cell), quantity, axis))
    _precision_mode = current_precision_mode()
    _level = (_conditioned or {}).get("level", "medium" if _precision_mode == "mixed" else "strict")
    _policy_id = (_conditioned or {}).get("policy_id", f"CMP-ZONE-ALLCLOSE-{_level.upper()}-001")
    comparison = ComparisonBinding(
        binding_id=f"DERIV-MATRIX-COMPARISON::{obligation_id}",
        policy_id=_policy_id,
        precision_mode=_precision_mode,
        zones=_COMPARISON_ZONES,
        level=_level,
        actual_role="xla-analytical",
        reference_role="xla-autodiff-same-precision",
    )
    input_binding.validate()
    electron_configuration.validate()
    comparison.validate()
    if ao not in _ao_families():
        raise AssertionError(f"matrix binding references unknown AO family: {ao}")
    return MatrixCaseBinding(
        guarantee_id=guarantee_id,
        obligation_id=obligation_id,
        pytest_node_id=pytest_node_id,
        input_binding=input_binding,
        electron_configuration=electron_configuration,
        comparison=comparison,
    )


_RAISE_CELLS: list[tuple] = []  # (cell, quantity, axis, why) -- analytic raised NotImplementedError = mission remainder
_NO_BUILDER: set[tuple] = set()  # cells with no committed builder path (fixture debt)
_FINDINGS: list[tuple] = []  # (cell, quantity, axis, why) -- analytic ran but diverged from _auto (a != b)


@pytest.fixture(autouse=True)
def _clear_bridge_caches():
    """Per-cell teardown: drop the UNBOUNDED bridge inner-jit LRU + XLA programs so walking
    dozens of distinct hams in one worker does not exhaust LLVM mmap (measured).
    """
    yield
    bridge.cache_clear()


def make_inventory_report(matrix, label):
    """Build a session-scoped autouse fixture that prints the ``label`` split-file's generated-cell
    header + the shared RAISE / NO-BUILDER / FINDINGS inventory (accumulated in this module's
    lists by ``run_matrix_cell``). Each split file instantiates one; the assertions live in
    ``run_matrix_cell``, so this is diagnostic-only.
    """

    @pytest.fixture(scope="session", autouse=True)
    def _matrix_report():
        n_light = sum(1 for _c, h in matrix if not h)
        n_heavy = sum(1 for _c, h in matrix if h)
        n_gem = {g: sum(1 for c, _h in matrix if _split_cell(c)[4] == g) for g in _GEMS}
        print(
            f"\n[matrix:{label}] generated {len(matrix)} cells "
            f"({n_light} light / {n_heavy} heavy) over "
            f"{len(_ao_families())} AO x {len(_jastrow_types())} Jastrow x {len(_ECPS)} ECP x {len(_BCS)} BC "
            f"x {len(_GEMS)} gem; gem counts={n_gem}; "
            f"sha256={_matrix_hash(matrix)[:16]}"
        )
        yield
        lines = ["", "=" * 80, f"Generated-matrix inventory [{label}] (this xdist worker; -rx aggregates all):", "-" * 80]
        lines.append(f"RAISE cells (analytic NotImplementedError) = remaining analytic gaps [{len(_RAISE_CELLS)}]:")
        for cell, q, ax, why in sorted(set(_RAISE_CELLS)):
            lines.append(f"  RAISE: {_cell_id(cell):40s} {q:7s} {ax:4s} :: {why}")
        lines.append(f"NO-BUILDER cells (fixture debt) [{len(_NO_BUILDER)}]:")
        for cell in sorted(_NO_BUILDER):
            lines.append(f"  NO-BUILDER: {_cell_id(cell):40s} :: {_no_builder_reason(cell)}")
        lines.append(f"FINDINGS (analytic ran but diverged: a != b) [{len(_FINDINGS)}]:")
        for cell, q, ax, why in sorted(set(_FINDINGS)):
            lines.append(f"  FINDING: {_cell_id(cell):40s} {q:7s} {ax:4s} :: {why}")
        lines.append("=" * 80)
        print("\n".join(lines))

    return _matrix_report


def run_matrix_cell(cell, quantity, axis):
    """Route-fidelity on one generated cell x (quantity, axis) -- OBSERVED, no capability gate.

    The analytic path is called directly (switch-ON) through the SAME production bridge
    factory (jit(vmap) over a walker batch, traced ham, running-SM geminal_inv) used by the
    ``_auto`` twin (switch-OFF):

    * a ``NotImplementedError`` is a mission-remainder RAISE cell (recorded, then re-raised
      as a hard failure -- RED, not xfail).
    * a successful run must equal the ``_auto`` twin (a == b); a divergence is a FINDING
      (recorded, then re-raised as a hard ``AssertionError`` failure -- RED, not xfail).
    """
    cell = tuple(cell)
    binding = matrix_case_binding(cell, quantity, axis)
    build = _resolve(cell)
    if build is None:
        _NO_BUILDER.add(cell)
        pytest.skip(f"no builder for cell {_cell_id(cell)}: {_no_builder_reason(cell)}")
    stem, with_jastrow, spec, gem = build

    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem)
    if _restores_bspline_j3_amplitude(stem, spec):
        ham = _restore_bspline_j3_amplitude(ham)
    r_up, r_dn = _draw_batch(ham)
    electron_receipt = observe_electron_configurations(binding.electron_configuration, r_up, r_dn)
    ginv = _batched_geminal_inv(ham, r_up, r_dn)

    mask = _flags_for_axis("c") if axis == "c" else None

    if quantity == "ln_Psi":
        make = lambda an: bridge.batched_grad_ln_Psi(wrt=axis, mask=mask, with_geminal_inv=True, analytic=an)  # noqa: E731
        call = lambda f: f(ham, r_up, r_dn, ginv)  # noqa: E731
    else:
        RT = jnp.broadcast_to(jnp.eye(3), (_NW, 3, 3))
        make = lambda an: bridge.batched_grad_e_L(  # noqa: E731
            wrt=axis, mask=mask, with_geminal_inv=True, analytic=an
        )
        call = lambda f: f(ham, r_up, r_dn, RT, ginv)  # noqa: E731

    # Direct call, no capability-gate pre-check: an unimplemented (quantity, axis) surfaces
    # as a real NotImplementedError, recorded as a RAISE cell (mission remainder) and
    # re-raised so pytest reports it RED, not xfail.
    capture = ProductionRouteCapture(
        _MATRIX_PRODUCTION_ROUTE,
        lambda args, kwargs: [(args[1], args[2])],
        extract_hamiltonian=lambda args, kwargs: args[0],
    )
    try:
        a = call(capture.wrap(make(True)))
    except NotImplementedError as e:
        why = str(e).splitlines()[0]
        _RAISE_CELLS.append((cell, quantity, axis, f"analytic RAISED: {why}"))
        raise

    # Analytic path ran -> it must equal the _auto twin in the same precision
    # lane (hard assert, no xfail escape). Both sides consume the same
    # Hamiltonian, electrons, precision policy, mask, and running inverse.
    b = call(make(False))
    try:
        compared_arrays = _assert_axis_allclose(
            ham, axis, a, b, cell_id=_cell_id(cell), quantity=quantity, level=binding.comparison.level
        )
    except AssertionError as e:
        why = str(e).splitlines()[0]
        _FINDINGS.append((cell, quantity, axis, f"a!=b: {why}"))
        raise

    _cond = CONDITIONED_CELLS.get((_cell_id(cell), quantity, axis))
    if _cond is not None:
        # VM-REGIME-CONDITIONED-PROOF-001: the reviewed numerical precondition is
        # OBSERVED in this same node, not asserted once in a comment.
        _lam = np.asarray(ham.wavefunction_data.geminal_data.lambda_matrix)
        _cond_lambda = float(np.linalg.cond(_lam))
        assert _cond["min_cond_lambda"] < _cond_lambda < _cond["max_cond_lambda"], (
            f"conditioned-proof precondition not met: cond_2(lambda) = {_cond_lambda:.3e} is outside the "
            f"reviewed band ({_cond['min_cond_lambda']:.0e}, {_cond['max_cond_lambda']:.0e}) for "
            f"{_cell_id(cell)} {quantity}/{axis}. The medium policy is justified by THIS conditioning; "
            "re-review the declaration."
        )
        # A conditioned proof must stay NECESSARY: if the strict policy now passes, the
        # declaration is stale and the cell must go back to strict.
        _sa, _sr = get_tolerance_min(binding.comparison.zones, "strict")
        _strict_fails = any(not np.allclose(np.asarray(x), np.asarray(y), atol=_sa, rtol=_sr) for x, y in compared_arrays)
        assert _strict_fails, (
            f"stale conditioned-proof declaration: {_cell_id(cell)} {quantity}/{axis} now meets the strict "
            "policy. Remove it from CONDITIONED_CELLS so the cell is proved at full strength again."
        )

    atol, rtol = get_tolerance_min(binding.comparison.zones, binding.comparison.level)
    comparison_receipt = observe_allclose(
        binding.comparison, compared_arrays, atol=atol, rtol=rtol, electron_receipt=electron_receipt
    )
    receipt = CaseReceipt(
        guarantee_id=binding.guarantee_id,
        obligation_id=binding.obligation_id,
        pytest_node_id=binding.pytest_node_id,
        input_binding=binding.input_binding,
        electron_configuration=electron_receipt,
        comparison=comparison_receipt,
        status="passed",
        route_capture=capture.receipt(),
    )
    receipt.validate()
    validate_matrix_fixture_route(receipt)
    from gdd.engine.evidence import emit_completion_record

    emit_completion_record(receipt)
    return receipt

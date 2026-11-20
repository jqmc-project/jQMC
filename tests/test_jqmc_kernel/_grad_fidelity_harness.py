"""The "8 sets" analytic-vs-``_auto`` integration harness +
a gap inventory.

The "8 sets" are ``{ln_Psi, e_L} x {c, r, R, eps}``, exercised through the PRODUCTION
bridge closures (``derivatives/bridge.py``) -- the SAME ``jit(vmap(...))``-over-a-
TRACED-ham factories the MCMC/GFMC drivers build (``jqmc_mcmc._cached_grad_*``), so
the harness is ROUTE-FAITHFUL: traced-ham + vmap-over-a-walker-batch + running-SM
``geminal_inv`` threading, exactly the production shape::

    a = bridge.batched_grad_<q>(wrt=axis, ..., analytic=True )(ham, r_up, r_dn[, RT], ginv)
    b = bridge.batched_grad_<q>(wrt=axis, ..., analytic=False)(ham, r_up, r_dn[, RT], ginv)
    tree-allclose(a, b)          # zone-derived tol (get_tolerance_min over crossed zones)

``a == b`` is switch-ON (``analytic=True``, the assembled analytic path) vs switch-OFF
(``analytic=False``, the ``_auto`` autograd twin) THROUGH THE SAME FACTORY -- both get
the identical ``wrt``/``mask``/``geminal_inv``, so the comparison is
apples-to-apples: whatever branches are frozen/pruned are frozen on both. ``wrt`` selects
the target axis -- for the c-axis the ``_C_MASK`` (every parameter block on EXCEPT the two
geminal radial-basis blocks, covered by focused basis and joint-MO/basis tests;
see ``_C_MASK``); for ``r``/``R``/``eps`` the physical-axis switch. Running this through the
production ``jit(vmap)`` bridge (not the eager public entry) is ALSO what keeps the harness
off the eager-microcompile-flood XLA-accumulation crash (the eager per-axis path floods
thousands of tiny compiles per call). Per-fixture ``bridge.cache_clear()`` teardown drops
the inner-jit LRU + XLA programs so walking 13 distinct hams does not exhaust LLVM mmap.

``analytic=True`` is the assembled analytic path; it raises ``NotImplementedError``
on a flagged axis it does not implement (NO silent ``_auto`` fallback).
This harness does NOT fill any analytic gap itself: where the analytic path raises,
the ``(fixture, quantity, axis)`` combination is recorded in the GAP inventory and the
test FAILS hard (RED, not xfail -- an unimplemented axis is a mission remainder for the
orchestrator to implement, not something to hide). The session-scoped
``_gap_inventory_report`` fixture prints the accumulated gaps at teardown, giving the
complete remaining-gap inventory across xdist workers.

Precision: the two sides are INDEPENDENT expression trees of the same mathematical
derivative (closed form vs autograd chain rule). Full mode evaluates both under the
full policy at the central strict tier. Mixed mode evaluates both under the same
mixed policy at the central medium tier. No derivative correctness receipt compares
results produced under different precision policies. The flag-splitting/contract
test is not precision-sensitive.
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
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine.derivative_inactivity import load_derivative_inactivity

pytestmark = [pytest.mark.with_chdir]

from jqmc import derivatives
from jqmc._setting import RTOL_PARAM_BLOCK_NONVACUITY, get_tolerance_mixed_fidelity
from jqmc._precision import get_tolerance_min
from jqmc.derivatives import bridge
from jqmc.determinant import Geminal_data
from jqmc.derivatives.bridge import _reduce_c_param_grads
from jqmc.jqmc_mcmc import _geminal_inv_batched
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._test_data_cache import qe_example_files_root
from tests._ecp_core_electrons import core_electrons_of
from tests._precision_fidelity import mixed_fidelity_blocks, out_of_budget_mixed_block
from tests.test_jqmc_kernel._gdd_adapters import current_precision_mode

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

# --------------------------------------------------------------------------- #
# QE pgto+pw COMPOSITE source (Li_gamma) for the dense-AGP fixtures.
#
# Forcing the bare periodic-pgto lithium basis into a dense AO-lambda AGP is
# ill-conditioned: that basis's near-singular overlap (min eig < 0 from image
# truncation) makes the AGP natural-orbital SVD ill-conditioned (SVD-at-near-
# degeneracy: an explicit singular-vector VJP and jax.grad genuinely disagree
# there). The dense-AGP fixtures therefore use a pGTO + plane-
# wave COMPOSITE built at test time from the committed Quantum-Espresso output
# (tests/qe_example_files/Li_gamma), via the same projection pipeline the light
# test_qe_to_jqmc.py tests exercise. Conditioning by construction:
#   * _QE_PGTO_ALPHA_WINDOW keeps only pGTO primitives the composite can support:
#     the lower edge drops diffuse primitives that the PW block ALREADY spans
#     exactly (a periodic pGTO's Fourier content lives on the same reciprocal
#     lattice as the PW child, so a low-alpha pGTO is a near-linear-dependence);
#     the upper edge drops the UPF ladder's ultra-tight primitive (alpha ~ 190),
#     which no cell-domain quadrature/PW grid resolves (its sampled norm ~ 0
#     would re-create an exact SVD zero).
#   * _QE_PW_G_CUT_FRACTION picks the low-|G| PW block (completeness) small enough
#     to keep the fixture light (~10 pGTO + ~33 PW AOs).
# Measured on this configuration: composite overlap eig in [1.9e-2, 2.9e+2]
# (cond ~ 1.6e4, strictly PSD) and, after the full-rank lift below, ALL natural-
# orbital singular values distinct (min |s_i^2 - s_j^2| ~ 5.7e-3, far above the
# _svd_vjp degen_tol 1e-10) -- a well-defined a==b compare on every axis, incl.
# the dense-AGP e_L/c that the bare-pgto basis could not reference.
# --------------------------------------------------------------------------- #
_QE_SAVE = os.path.join(qe_example_files_root(), "Li_gamma", "out", "li.save")
_QE_STEM = "QE:Li_gamma_pgto+pw"  # sentinel stem routed to _qe_composite_pieces by _build_ham
_QE_AVAILABLE = all(os.path.isfile(os.path.join(_QE_SAVE, f)) for f in ("data-file-schema.xml", "wfc1.hdf5", "Li.ccECP.upf"))
requires_qe = pytest.mark.skipif(not _QE_AVAILABLE, reason="QE Li_gamma sample not present")
_QE_PGTO_ALPHA_WINDOW = (1.0, 8.0)  # keep pGTO primitives resolvable AND independent of the PW block
_QE_PW_G_CUT_FRACTION = 0.25  # PW block: |G| <= fraction * g_max(ecutwfc)
_AGP_LIFT_PERT_SCALE = 0.1  # symmetric random perturbation of the lifted AGP lambda
_AGP_LIFT_DIAG_RAMP = 0.2  # strictly-distinct diagonal ramp step (separates the natural-orbital SVs)


def _window_shells(shells_per_atom, lo: float, hi: float):
    """Keep only the pGTO primitives with ``lo <= exponent <= hi`` (see the conditioning
    note above); shells left empty are dropped."""
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


@functools.lru_cache(maxsize=1)
def _qe_composite_pieces():
    """(structure, composite AO leaf, MO geminal, ECP coulomb) from the committed Li_gamma QE
    output, built ONCE per process (the projection + Ewald build is a few seconds). Mirrors the
    light test_qe_to_jqmc.py / test_integration_qe_* assembly: load the QE reference, project the
    occupied bands onto the windowed-pGTO + low-G-PW mixed basis, and wrap with the full analytic
    ccECP (local + non-local) parsed from the UPF [Gaussinfo] block."""
    from jqmc.pw_pgto_projection import project_pw_to_mixed
    from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit, parse_gaussinfo

    upf = os.path.join(_QE_SAVE, "Li.ccECP.upf")
    ref = load_qe_reference(_QE_SAVE)
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    strat = UPFExponentInit({"li": upf}, g_target=g_max)
    shells = _window_shells([strat.shells_for_element(lbl) for lbl in ref.atom_labels], *_QE_PGTO_ALPHA_WINDOW)
    proj = project_pw_to_mixed(ref, shells, g_cut=_QE_PW_G_CUT_FRACTION * g_max, drop_diffuse=False)
    structure_data, aos_data, geminal_mo_data = build_wavefunction_from_projection(ref, proj, kind="pgto+pw")
    ham = build_hamiltonian_from_wavefunction(
        structure_data,
        aos_data,
        geminal_mo_data,
        j1_param=None,
        j2_param=None,
        ecp_by_element={"Li": parse_gaussinfo(upf)},
    )
    return structure_data, aos_data, geminal_mo_data, ham.coulomb_potential_data


# The 11 parameter blocks == the ``with_param_grad_mask`` keyword set. The
# c-axis is turned on by flagging parameter blocks True (there is no opt_c).
_PARAM_BLOCKS = (
    "opt_J1_param",
    "opt_J2_param",
    "opt_J3_param",
    "opt_geminal",
    "opt_J3_basis_exp",
    "opt_J3_basis_coeff",
    "opt_lambda_basis_exp",
    "opt_lambda_basis_coeff",
    "opt_JDTN_u_param",
    "opt_JDTN_chi_param",
    "opt_JDTN_f_param",
)

# Keep the reviewed matrix's non-basis c-axis scope unchanged. Geminal radial
# blocks have focused basis-only and joint-MO/basis evidence. These flags are
# independent in production; their separation here is a coverage partition,
# not a prohibition on simultaneous optimization. J3 basis blocks stay on.
_C_MASK = {k: ("lambda_basis" not in k) for k in _PARAM_BLOCKS}


def _flags_for_axis(axis: str) -> dict:
    """Unified ``opt_flags`` selecting exactly one target axis.

    ``c`` -> the c-axis mask ``_C_MASK`` (all parameter blocks True except the two geminal
    radial-basis blocks; see ``_C_MASK`` for why they are excluded). ``r``/``R``/``eps`` -> the single
    physical-axis switch. Both the analytic and ``_auto`` paths receive this same dict, so any frozen
    basis block is frozen symmetrically.
    """
    if axis == "c":
        return dict(_C_MASK)
    return {f"opt_{axis}": True}


# --- fixture matrix (LIGHT): molecular cart / sphe / ECP GTO + light periodic
# (real Gamma & complex twist), with and without an analytic-polynomial Jastrow.
# (id, trexio stem, with_jastrow, spec). ``spec`` is an optional dict extending the
# wavefunction so the c-axis exercises the two known remaining analytic gaps that the
# base matrix (J1+J2, MO/SD geminal) does NOT hit:
#   * ``"j3"``: attach an ACTIVE three-body Jastrow whose orbital leaf is of the given
#     type -- ``"pgto"`` / ``"complex"`` (reuse the fixture's own AO leaf) or ``"mo"``
#     (that AO leaf wrapped in an ``MOs_data``). The c-axis mask turns on
#     ``opt_J3_basis_exp/coeff``; that block is analytic ONLY for a single-leaf REAL
#     cart/sphe GTO J3 orbital (assemble/_capability), so a pgto / complex-twist / MO
#     J3 orbital surfaces gap #33 (J3-basis on a non-cart/sphe-GTO leaf).
#   * ``"dense_agp"``: force the MO/SD geminal into a dense-lambda AO-AGP
#     (``convert_from_MOs_to_AOs``) -- the #35 AO-AGP shape.
# Reuses the committed trexio sources -- no h5. ---
_FIXTURES = [
    ("H2_cart_J", "H2_ae_ccpvdz_cart.h5", True, None),  # molecular cart-GTO, closed-shell, J1+J2
    ("H2_cart_noJ", "H2_ae_ccpvdz_cart.h5", False, None),  # ... no Jastrow
    ("N_cart_J", "N_ae_ccpvdz_cart.h5", True, None),  # molecular cart-GTO, OPEN-shell, J1+J2
    ("H2_sphe_J", "H2_ae_ccpvdz_sphe.h5", True, None),  # molecular sphe-GTO
    ("H2_ecp_cart_J", "H2_ecp_ccpvdz_cart.h5", True, None),  # molecular ECP (local + non-local)
    ("H2_ecp_cart_noJ", "H2_ecp_ccpvdz_cart.h5", False, None),  # ... no Jastrow
    ("Li_pbc_gamma_J", "lithium.h5", True, None),  # periodic real Gamma, ECP sphe-pgto, J1+J2
    ("Li_pbc_twist_J", "lithium_k025.h5", True, None),  # periodic complex twist (k != 0)
    # --- complex-twist + DTN polynomial Jastrow (u/chi/f). The twist x DTN x ECP x e_L/c cell was a
    # gate over-claim (dtn.grad_T_dtn_wrt_param complex scan-carry / species-scatter dtypes), fixed
    # by result_type(M, GD); this row exercises that cell PLUS the DTN physical axes on twist.
    ("Li_pbc_twist_JDTN", "lithium_k025.h5", True, {"dtn": True}),
    # --- #33: J3-basis optimization beyond the single-leaf cart/sphe GTO orbital -----
    # pGTO J3 orbital (periodic real Gamma) and complex-twist pGTO J3 orbital: c-axis
    # opt_J3_basis + all physical axes analytic (filled #33 subcases).
    ("Li_pbc_gamma_J3pgto", "lithium.h5", True, {"j3": "pgto"}),
    ("Li_pbc_twist_J3cplx", "lithium_k025.h5", True, {"j3": "complex"}),
    # --- MO-backed J3 on a PLAIN (non dense-AO-AGP) molecular geminal ------------------
    # The default molecular geminal is a plain SD (MO leaves); an MO-backed J3 (cart-GTO leaf
    # wrapped in an identity MOs_data) exercises the MO->bare-AO J3 fold (_j3_mo_to_bare_ao)
    # on the R / eps (force / stress) axes for a geminal that is NOT the dense-AO-AGP pairing.
    # The fold is geminal-independent, so all 8 sets
    # (ln_Psi/e_L x c/r/R/eps) must match _auto -- guards against re-refusing MO-J3 Pulay.
    ("H2_SD_J3MO", "H2_ae_ccpvdz_cart.h5", True, {"j3": "mo"}),
    # --- dense AO-AGP on the QE pgto+pw COMPOSITE (well-conditioned; see the note at
    # _qe_composite_pieces). MO-backed J3 whose underlying AO leaf is the COMPOSITE itself:
    # closes the last #33 subcase (composite J3-basis: value O_k + kinetic dT/dc contract
    # per child; PW children carry no radial params) AND the composite J3 physical axes
    # (drift / force / stress descend to the children, PW rows R-independent /
    # affine-value-invariant). Paired with the dense AO-AGP geminal (the MO-J3 R/eps fold
    # ``_j3_mo_to_bare_ao`` is verified for the dense-AGP pairing only). The c-axis compare
    # covers the per-child flat J3 basis blocks, and the raw child leaves are independently
    # verified in test_composite_j3_basis_grads_vs_auto below.
    pytest.param("Li_AGP_J3MO", _QE_STEM, True, {"dense_agp": True, "j3": "mo"}, marks=requires_qe),
]
_FIXTURE_IDS = [(f.values[0] if hasattr(f, "values") else f[0]) for f in _FIXTURES]


def _j3_orbital(aos_data, kind: str):
    """The J3 orbital leaf for the requested ``kind``: the AO leaf itself (``"pgto"`` /
    ``"complex"``) or that leaf wrapped in an identity ``MOs_data`` (``"mo"``)."""
    if kind == "mo":
        return MOs_data(num_mo=aos_data.num_ao, aos_data=aos_data, mo_coefficients=np.eye(aos_data.num_ao))
    return aos_data


def _make_j3(orb_data) -> Jastrow_three_body_data:
    """An ACTIVE three-body Jastrow on ``orb_data`` (an AO leaf or an ``MOs_data``).

    Random symmetric coupling block + a linear final column so every J3 term is nonzero
    (mirrors ``test_jqmc_derivatives._build_ham_j3``). The J3 ORBITAL TYPE is what the
    c-axis ``opt_J3_basis`` capability keys on -- a pgto / complex-twist / MO leaf keeps
    that block on ``_auto`` (gap #33), a single-leaf real cart/sphe GTO leaf is analytic.
    """
    rng = np.random.default_rng(5)
    n = int(orb_data._num_orb)
    jm = rng.normal(size=(n, n + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    from jqmc.jastrow_factor import _phase_free_j3_orbitals

    return Jastrow_three_body_data(orb_data=_phase_free_j3_orbitals(orb_data), j_matrix=jm)


def _make_dtn(structure_data, coulomb_potential_data) -> Jastrow_DTN_data:
    """An ACTIVE DTN polynomial Jastrow (u/chi/f channels; SMALL orders to keep the row light).

    ``use_f=True`` is required so the f-channel (e-e-N) kinetic ``dT/dc`` scan -- the site of the
    complex-twist scan-carry dtype bug -- is exercised. Mirrors the in-test build in
    ``test_complex_twist_dtn_caxis._build_twist_dtn_ham``.
    """
    atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.float64)
    z_cores = (
        np.asarray(coulomb_potential_data.z_cores, dtype=np.float64)
        if coulomb_potential_data.ecp_flag
        else np.zeros_like(atomic_numbers)
    )
    z_eff_per_species = tuple(
        float(np.mean((atomic_numbers - z_cores)[atomic_numbers == Z])) for Z in np.unique(atomic_numbers)
    )
    pseudized_per_species = tuple(
        bool(np.all(coulomb_potential_data.pseudized_per_atom[atomic_numbers == Z])) for Z in np.unique(atomic_numbers)
    )
    return Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff_per_species,
        pseudized_per_species=pseudized_per_species,
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


def _make_full_rank_ao_agp(geminal_mo_data) -> Geminal_data:
    """A GENUINE, well-conditioned full-rank dense AO-lambda AGP (gap #35 shape).

    ``Geminal_data.convert_from_MOs_to_AOs`` yields a RANK-DEFICIENT lambda (rank == n_occ; e.g. rank-1
    for the 1up/1dn lithium fixture) whose biorthogonal ``f = S_up^{1/2} lambda S_dn^{1/2}`` has
    EXACTLY-degenerate ZERO singular values, so the natural-orbital SVD used by both the analytic
    explicit singular-vector VJP (its 1/(s_j^2 - s_i^2)) AND ``jax.grad`` (its own SVD
    backward) is non-differentiable there -- an intrinsic SVD-at-degeneracy singularity, not an
    analytic gap. Lift the converted lambda to a legitimate full-rank AGP by adding a symmetric
    perturbation with a strictly-distinct (ramped) diagonal to the paired block, so ALL singular values
    of ``f`` are distinct and nonzero and the AO-AGP c-axis (e_L/c) is a well-defined a==b compare.
    Still a dense AO-AGP (raw AO leaves, not ``MOs_data``); only the lambda numbers change.

    COMPLEX-TWIST (k != 0): ``convert_from_MOs_to_AOs`` yields a COMPLEX HERMITIAN lambda there
    (``lambda == lambda^H``; measured Hermitian residual ~3e-17, complex-symmetric residual ~59 on the
    H_k025 composite). The complex branch keeps the complex dtype and mirrors the real lift with a
    HERMITIAN symmetrization (``0.5 * (pert + pert^H)``) and a REAL distinct-diagonal ramp (a Hermitian
    diagonal is real, so a real ramp preserves ``lambda == lambda^H``). This keeps ``f`` Hermitian with
    strictly-distinct nonzero singular values so the natural-orbital SVD backward stays differentiable.
    The REAL path is preserved byte-for-byte (only the ``is_complex`` branch differs).
    """
    agp = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    lam_src = np.asarray(agp.lambda_matrix)  # (orb_num_up, orb_num_dn[+unpaired])
    is_complex = np.iscomplexobj(lam_src)
    lam = np.array(lam_src, dtype=np.complex128 if is_complex else np.float64)
    n_dn = int(agp.orb_num_dn)
    paired = lam[:, :n_dn]  # square for lithium (n_up == n_dn)
    n_up, n_cols = paired.shape
    rng = np.random.default_rng(7)
    if is_complex:
        pert = _AGP_LIFT_PERT_SCALE * (rng.normal(size=(n_up, n_cols)) + 1j * rng.normal(size=(n_up, n_cols)))
        if n_up == n_cols:  # Hermitian perturbation keeps a clean same-basis complex-twist AGP (lambda == lambda^H)
            pert = 0.5 * (pert + pert.conj().T)
    else:
        pert = _AGP_LIFT_PERT_SCALE * rng.normal(size=(n_up, n_cols))
        if n_up == n_cols:  # symmetric perturbation keeps a clean same-basis AGP
            pert = 0.5 * (pert + pert.T)
    k = min(n_up, n_cols)  # strictly-distinct diagonal ramp => no accidental singular-value degeneracy
    pert[np.arange(k), np.arange(k)] += _AGP_LIFT_DIAG_RAMP * (1.0 + np.arange(k))
    lam[:, :n_cols] = paired + pert
    return agp.replace(lambda_matrix=lam)


def _build_ham(stem: str, with_jastrow: bool, spec: dict | None = None) -> Hamiltonian_data:
    spec = spec or {}
    if stem == _QE_STEM:  # QE pgto+pw composite source (cached; see _qe_composite_pieces)
        structure_data, aos_data, geminal_mo_data, coulomb_potential_data = _qe_composite_pieces()
    else:
        (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
            os.path.join(_TREXIO, stem), store_tuple=True
        )
    # geminal: MO/SD (default) or forced dense-lambda AO-AGP (gap #35 shape). The dense-AGP fixtures use
    # a GENUINE full-rank AGP (``_make_full_rank_ao_agp``); a bare ``convert_from_MOs_to_AOs`` is
    # rank-deficient (SVD-degenerate => non-differentiable, so a==b would be ill-defined).
    geminal_data = _make_full_rank_ao_agp(geminal_mo_data) if spec.get("dense_agp") else geminal_mo_data
    # optional active three-body Jastrow whose orbital leaf carries the #33 gap.
    j3 = _make_j3(_j3_orbital(aos_data, spec["j3"])) if spec.get("j3") else None
    # optional active DTN polynomial Jastrow (u/chi/f); the twist+DTN c-axis was a gate over-claim
    # (dtn.grad_T_dtn_wrt_param complex scan-carry / scatter dtype), now fixed.
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
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


def _draw(ham: Hamiltonian_data, seed: int = 0):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(nu, 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(nd, 3))),
    )


# Walker batch size for the route-faithful compare: the production bridge closures are
# ``jit(vmap(...))`` over a walker batch, so the harness feeds a small (nw=2) batch and a
# matching per-walker running-SM ``geminal_inv`` -- exactly the driver's per-step shape.
_NW = 2


def _draw_batch(ham: Hamiltonian_data, nw: int = _NW, seed: int = 0):
    """A ``(nw, N, 3)`` walker batch of electron positions (the production per-step batch shape)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    # GDD records the concrete bit generator. Do not rely on NumPy's movable
    # ``default_rng`` alias for a certifying electron-configuration sequence.
    rng = np.random.Generator(np.random.PCG64(seed))
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(nw, nu, 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(nw, nd, 3))),
    )


def _batched_geminal_inv(ham: Hamiltonian_data, r_up_b, r_dn_b):
    """A REAL batched running-SM geminal inverse (``geminal_pinv`` of the batch), threaded into the
    ``with_geminal_inv=True`` bridge closures -- the production shape (the driver reuses the running
    Sherman-Morrison inverse so no fresh SVD-pinv lowers). ``_auto`` ignores it and recomputes the
    inverse internally; the analytic path consumes it -- both are the same fp64 inverse, so a==b holds."""
    return _geminal_inv_batched(ham.wavefunction_data.geminal_data, r_up_b, r_dn_b)[1]


@pytest.fixture(autouse=True)
def _clear_bridge_caches():
    """Per-test teardown: drop the bridge inner-jit LRU caches + XLA compiled programs. The bridge
    caches are UNBOUNDED and this module walks 13 distinct hams; without the clear the accumulated
    compiled programs exhaust LLVM mmap (measured). Production never needs it (one ham per run)."""
    yield
    bridge.cache_clear()


# --------------------------------------------------------------------------- #
# GAP inventory: (fixture, quantity, axis) combinations whose analytic=True path
# still raises NotImplementedError. Accumulated per xdist worker and printed by the
# session-scoped fixture below (each such combination is also a hard test FAILURE,
# not an xfail -- see ``test_8set_analytic_vs_auto``).
# --------------------------------------------------------------------------- #
_GAPS: list[tuple[str, str, str, str]] = []


def _record_gap(fixture_id: str, quantity: str, axis: str, reason: str) -> None:
    _GAPS.append((fixture_id, quantity, axis, reason.splitlines()[0]))


@pytest.fixture(scope="session", autouse=True)
def _gap_inventory_report():
    yield
    if _GAPS:
        lines = [
            "",
            "=" * 78,
            "analytic-gradient GAP inventory -- (fixture, quantity, axis) still raising",
            "NotImplementedError on analytic=True (this xdist worker; -rx aggregates all):",
            "-" * 78,
        ]
        for fid, q, ax, why in sorted(_GAPS):
            lines.append(f"  GAP: {fid:20s} {q:7s} {ax:4s} :: {why}")
        lines.append("=" * 78)
        print("\n".join(lines))


# Cells that provably carry NO evidence for a parameter block, declared rather than
# hidden behind a passing comparison. Measured ratios (block peak / cell scale) sit
# at 1e-10 .. 1e-17 against 1e-2 for the healthy leaves: on these non-GTO leaves the
# whitened AGP_MO lambda lands in directions the basis barely spans, so e_L does not
# see it. Deleting the analytic lambda derivative keeps every one of these green --
# that is what makes the comparison vacuous rather than merely tight.
#
# Keyed by QUANTITY as well as block, because the collapse is specific to e_L: on the
# same cells ``dlnPsi/dlambda`` peaks at 5.5e+01 .. 3.3e+02 (healthy) while
# ``de_L/dlambda`` sits at 1e-9. That is the signature of lambda acting mostly through
# the overall normalization of Psi, which lnPsi sees and the local energy -- a ratio
# HPsi/Psi -- is invariant to. The first draft of this set omitted the quantity and the
# stale-declaration check below rejected it immediately.
#
# This set is checked in BOTH directions: a declared cell that regains sensitivity
# fails as a stale declaration. Entries leave only by fixing the fixture.
# Currently EMPTY: the matrix does not admit the whitened AGP_MO cells on the QE-projection
# leaves at all (gdd.engine.derivative_matrix._agp_mo_constructible). The screen and the
# declaration machinery serve the next fixture that cannot see its own parameter: it must be
# declared or fixed, not quietly certified.
NO_PARAM_EVIDENCE: frozenset[tuple[str, str, str]] = frozenset()


# Exact reviewed non-live blocks remain IN the numerical comparison. The SSoT
# declaration does not certify their derivative implementation or waive a check;
# _assert_nonvacuous below rejects both undeclared inactivity and stale declarations.
STRUCTURALLY_ZERO_BLOCKS = load_derivative_inactivity()


def _assert_nonvacuous(cell_id, quantity, axis, named_references, atol, rtol) -> None:
    """Require every compared block / leaf to carry evidence, or a TRUE declaration.

    A compared array is LIVE when its reference (the independent ``_auto`` side) satisfies both
    (i) peak >= RTOL_PARAM_BLOCK_NONVACUITY x the cell scale (the largest reference peak among
    the arrays compared in this call), and
    (ii) the array is not within the comparison's own (atol, rtol) of zero, so an analytic
    block replaced by zeros would fail this very comparison.
    A non-live array must be declared in ``STRUCTURALLY_ZERO_BLOCKS``; a declared array that is
    live is a stale declaration. All
    violations of one call are reported together. Size-0 leaves (no electrons of one spin) have
    no components and are not screened.
    """
    named = [(name, ref) for name, ref in named_references if ref.size]
    peaks = {name: float(np.max(np.abs(ref))) for name, ref in named}
    scale = max(peaks.values(), default=0.0)
    problems = []
    for name, ref in named:
        near_zero = bool(np.allclose(np.zeros_like(ref), ref, atol=atol, rtol=rtol))
        relative = peaks[name] >= RTOL_PARAM_BLOCK_NONVACUITY * scale
        live = relative and not near_zero
        declaration = STRUCTURALLY_ZERO_BLOCKS.get((cell_id, quantity, axis, name))
        ratio = peaks[name] / scale if scale > 0.0 else 0.0
        observed = f"reference peaks at {peaks[name]:.3e} (cell scale {scale:.3e}, ratio {ratio:.2e})"
        if declaration is None:
            if not live:
                problems.append(f"vacuous {name}: {observed}")
            continue
        disposition = declaration[0]
        holds = not live
        if not holds:
            problems.append(
                f"stale {disposition} declaration {name}: {observed} is live; remove it from STRUCTURALLY_ZERO_BLOCKS"
            )
    assert not problems, (
        f"non-vacuity screen failed for cell={cell_id} quantity={quantity} axis={axis}: "
        + "; ".join(problems)
        + f". Threshold: ratio >= {RTOL_PARAM_BLOCK_NONVACUITY:.0e} and not allclose(0, ref, atol={atol:.1e}, "
        f"rtol={rtol:.1e}). A vacuous comparison would stay green if the analytic derivative were deleted: "
        "fix the fixture so the block is exercised, or declare it in STRUCTURALLY_ZERO_BLOCKS with the reason."
    )


def _assert_axis_allclose(
    ham: Hamiltonian_data, axis: str, a, b, cell_id: str | None = None, quantity: str | None = None, level: str = "strict"
):
    """Compare the analytical batch with its independent ``_auto`` oracle.

    Full compares both routes under the full policy at the strict tier. Mixed
    compares both routes under the mixed policy at the central medium tier.
    """
    # ``level`` comes from the case's ComparisonBinding, so the harness and the GDD receipt
    # resolve the SAME tolerance from the SAME central resolver -- a cell bound to the
    # conditioned-proof policy must not be judged strictly here and leniently in its receipt.
    resolved_level = "medium" if level == "strict" and current_precision_mode() == "mixed" else level
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
        resolved_level,
    )
    if axis == "c":  # variational-parameter blocks (O_k / dE_L/dc): compare PARAM BLOCKS, never raw
        # tree leaves (documented trap: the batched RAW ``lambda_matrix`` property hstacks a fixed 2-D
        # identity paired block with the walker-leading unpaired grad -> shape crash / spurious block).
        # Use production's ``_reduce_c_param_grads`` (the exact reduction the driver stores) on BOTH,
        # then production's downstream block filter: run() drops blocks whose opt flag is off
        # (jqmc_mcmc.py ``if not self.__param_grad_flags.get(name, True): continue``) -- needed because
        # the AGP_data reduce path (``Geminal_data.collect_param_grads``) emits ``lambda_basis_*``
        # MASK-BLIND whenever GTO leaves exist, while the analytic assemble computes basis grads
        # structurally (``_geminal_has_opt_basis``) and ``_auto`` honours the in-trace freeze (zeros):
        # without the filter the two sides compare an unopted block production never stores.
        wf = ham.wavefunction_data
        mask_kwargs = _flags_for_axis("c")  # the c-axis mask (``_C_MASK``); SAME mask both closures got
        block_flag = {
            "lambda_basis_exp": "opt_lambda_basis_exp",
            "lambda_basis_coeff": "opt_lambda_basis_coeff",
            "j3_basis_exp": "opt_J3_basis_exp",
            "j3_basis_coeff": "opt_J3_basis_coeff",
        }

        def _stored(blocks):
            return {k: v for k, v in blocks.items() if mask_kwargs.get(block_flag.get(k), True)}

        da = _stored(_reduce_c_param_grads(wf, a, _NW, mask_kwargs))
        db = _stored(_reduce_c_param_grads(wf, b, _NW, mask_kwargs))
        expected = {block.name for block in wf.get_variational_blocks(state_token=("test", 0), **mask_kwargs)}
        assert expected, "no variational blocks declared -- c-axis comparison would be vacuous"
        assert set(da) == expected, (
            f"analytical param-block contract differs: missing={expected - set(da)}, extra={set(da) - expected}"
        )
        assert set(db) == expected, (
            f"autograd param-block contract differs: missing={expected - set(db)}, extra={set(db) - expected}"
        )
        # Every block (real J1/J2/J3/DTN/basis AND the complex-twist geminal ``mo_coefficients_up``)
        # is compared with FULL complex tolerance -- Re AND Im. The complex-twist geminal-mo c-grad
        # Im channel (task #12) is now analytic-correct: assemble._set_geminal_mo_grad stores the two
        # spin channels on their own leaves and the tie ``C_dn = conj(C_up)`` is folded LATE at
        # ``_reduce_c_param_grads`` (``gu + conj(gd)``, matching ``_auto``), and compute_grad_e_L's
        # e_L/c Re-projection is per-block (keyed off the PARAMETER dtype) so the complex mo block
        # keeps its genuinely-complex cotangent while real blocks stay Re-projected.
        # Non-vacuity screen: see ``_assert_nonvacuous``. Every compared block is screened
        # (declared-no-evidence blocks are excluded from the comparison below and from it).
        _assert_nonvacuous(
            cell_id,
            quantity,
            axis,
            [(k, np.asarray(db[k])) for k in db if (cell_id, quantity, k) not in NO_PARAM_EVIDENCE],
            atol,
            rtol,
        )
        # A NO_PARAM_EVIDENCE declaration is not a licence: it must stay TRUE. If the
        # fixture regains sensitivity the declaration is stale and must be removed.
        _scale = max((float(np.max(np.abs(np.asarray(v)))) for v in da.values() if np.asarray(v).size), default=0.0)
        for k in da:
            # Screened only where a declaration exists (a selected suite, not a skipped loop tail).
            if (cell_id, quantity, k) in NO_PARAM_EVIDENCE and _scale > 0.0:
                _block = float(np.max(np.abs(np.asarray(da[k]))))
                assert _block < RTOL_PARAM_BLOCK_NONVACUITY * _scale, (
                    f"stale declaration: {cell_id} is declared to carry no {k} evidence, but the "
                    f"block now peaks at {_block:.3e} against a cell scale of {_scale:.3e} "
                    f"(ratio {_block / _scale:.2e}). Remove it from NO_PARAM_EVIDENCE."
                )

        compared_arrays = []
        for k in da:
            if (cell_id, quantity, k) in NO_PARAM_EVIDENCE:
                # Declared to carry no evidence: asserting agreement on it would be
                # asserting agreement on round-off. It is excluded from the comparison
                # as well as from the screen, so the declaration means one thing only --
                # this cell says nothing about this block -- rather than quietly
                # certifying noise.
                continue
            va, vb = np.asarray(da[k]), np.asarray(db[k])
            assert np.all(np.isfinite(va)), f"analytical block {k} contains NaN/Inf"
            assert np.all(np.isfinite(vb)), f"autograd block {k} contains NaN/Inf"
            np.testing.assert_allclose(va, vb, atol=atol, rtol=rtol, err_msg=f"block {k}")
            compared_arrays.append((va, vb))
        assert compared_arrays, "every parameter block was declared evidence-free: the cell would certify nothing"
        return tuple(compared_arrays)

    # r/R/eps: assembled axis grad (walker-leading), leaf-for-leaf
    la_, lb_ = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la_) == len(lb_), f"leaf count differs: {len(la_)} vs {len(lb_)}"
    leaf_names = [jax.tree_util.keystr(path) or "<root>" for path, _ in jax.tree_util.tree_flatten_with_path(b)[0]]
    _assert_nonvacuous(cell_id, quantity, axis, [(n, np.asarray(lb)) for n, lb in zip(leaf_names, lb_)], atol, rtol)
    compared_arrays = []
    for la, lb in zip(la_, lb_):
        va, vb = np.asarray(la), np.asarray(lb)
        np.testing.assert_allclose(va, vb, atol=atol, rtol=rtol)
        compared_arrays.append((va, vb))
    return tuple(compared_arrays)


@pytest.mark.light
@pytest.mark.parametrize("fixture_id,stem,with_jastrow,spec", _FIXTURES, ids=_FIXTURE_IDS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("axis", ["c", "r", "R", "eps"])
def test_8set_analytic_vs_auto(fixture_id, stem, with_jastrow, spec, quantity, axis):
    """One of the 8 sets on one fixture, ROUTE-FAITHFUL: switch-ON (analytic, assembled) == switch-OFF
    (``_auto`` autograd twin) THROUGH THE PRODUCTION BRIDGE -- ``jit(vmap)`` over a walker batch with a
    TRACED ham and running-SM ``geminal_inv`` threading -- exactly the driver's shape.

    Where analytic=True raises NotImplementedError (at trace time) the combination is a documented
    analytic gap -> recorded in the GAP inventory, then re-raised as a hard failure (RED, not xfail;
    it is a mission remainder for the orchestrator to implement, not something this harness hides).
    Both closures come from the SAME factory with the SAME ``wrt``/``mask``/``geminal_inv``,
    differing ONLY in ``analytic=True`` vs ``False`` (the a==b switch).
    """
    ham = _build_ham(stem, with_jastrow, spec)
    r_up, r_dn = _draw_batch(ham)
    ginv = _batched_geminal_inv(ham, r_up, r_dn)

    # a = switch-ON (analytic) closure; b = switch-OFF (_auto) closure -- SAME factory, SAME route.
    # c-axis: the ``_C_MASK`` (all blocks minus the conflicting geminal-basis pair); physical axes
    # ignore ``mask``. Both closures get the identical mask so any frozen block is frozen on both.
    c_mask = _flags_for_axis("c") if axis == "c" else None
    if quantity == "ln_Psi":
        make = lambda an: bridge.batched_grad_ln_Psi(wrt=axis, mask=c_mask, with_geminal_inv=True, analytic=an)  # noqa: E731
        call = lambda f: f(ham, r_up, r_dn, ginv)  # noqa: E731
    else:
        RT = jnp.broadcast_to(jnp.eye(3), (_NW, 3, 3))
        make = lambda an: bridge.batched_grad_e_L(  # noqa: E731
            wrt=axis, mask=c_mask, with_geminal_inv=True, analytic=an
        )
        call = lambda f: f(ham, r_up, r_dn, RT, ginv)  # noqa: E731

    # analytic path: NotImplementedError (raised during the jit trace) == a remaining analytic gap.
    # Record it in the GAP inventory for teardown reporting, then FAIL hard -- a mission remainder
    # the orchestrator must implement, not an xfail escape.
    try:
        a = call(make(True))
    except NotImplementedError as e:
        _record_gap(fixture_id, quantity, axis, str(e))
        pytest.fail(f"GAP {fixture_id} {quantity}/{axis}: analytic NotImplementedError -- {str(e).splitlines()[0]}")

    b = call(make(False))  # same-precision _auto oracle in both lanes
    _assert_axis_allclose(ham, axis, a, b, cell_id=fixture_id, quantity=quantity)


@pytest.mark.light
@requires_qe
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_composite_j3_basis_grads_vs_auto(quantity):
    """Composite J3 basis: the analytic per-child radial exp/coeff grads (value O_k via
    basis.grad_J3_wrt_basis + kinetic dT/dc via basis.grad_T_wrt_J3_basis, both contracting
    per composite child through basis._basis_contract_*_general / ao_deriv.j3_basis_derivs)
    equal the ``_auto`` autograd grads. The production flat block dict publishes each
    composite child as ``j3_child<i>_basis_*``; this orthogonal assertion also compares
    the RAW grad-pytree radial leaves (PW children carry no radial params)."""
    ham = _build_ham(_QE_STEM, True, {"dense_agp": True, "j3": "mo"})
    r_up, r_dn = _draw(ham)
    flags = _flags_for_axis("c")
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
        b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
        b = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)["c"]
    ga = a.jastrow_data.jastrow_three_body_data.orb_data.aos_data  # grad pytree: MO wrapper -> composite
    gb = b.jastrow_data.jastrow_three_body_data.orb_data.aos_data
    # weakest crossed zone (J3-basis radial params: ao + jastrow + det path); mixed -> fp32 floor.
    level = "medium" if current_precision_mode() == "mixed" else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), level)
    compared = 0
    for i, (ca, cb) in enumerate(zip(ga.children, gb.children)):
        if "gto" not in type(ca).__name__:
            continue
        np.testing.assert_allclose(
            np.asarray(ca.radial.exponents),
            np.asarray(cb.radial.exponents),
            atol=atol,
            rtol=rtol,
            err_msg=f"{quantity} composite J3-basis child{i} d/d exponents",
        )
        np.testing.assert_allclose(
            np.asarray(ca.radial.coefficients),
            np.asarray(cb.radial.coefficients),
            atol=atol,
            rtol=rtol,
            err_msg=f"{quantity} composite J3-basis child{i} d/d coefficients",
        )
        compared += 1
    assert compared > 0, "no GTO child compared -- the composite J3-basis check would be vacuous"


@pytest.mark.light
@pytest.mark.parametrize("axis", ["c", "r", "R", "eps"])
def test_flag_split_selects_requested_axis(axis):
    """Contract (no numeric compare -> not precision-sensitive): the unified interface
    returns a dict keyed exactly by the requested axis, the c-axis is implied by the
    parameter-block flags (no ``opt_c``), and the ``_auto`` path honours ``flags``."""
    ham = _build_ham("H2_ae_ccpvdz_cart.h5", with_jastrow=True)
    r_up, r_dn = _draw(ham)
    flags = _flags_for_axis(axis)

    out = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)
    assert set(out) == {axis}, f"{axis}: expected dict keyed by {{{axis!r}}}, got {sorted(out)}"
    outL = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)
    assert set(outL) == {axis}

    # c-axis is implied by any parameter block being True; the empty flag set selects nothing.
    assert set(derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {}, analytic=False)) == set()
    assert "opt_c" not in _flags_for_axis("c")


# Numerical fidelity helpers are separate from the unchanged mathematical AD
# comparator above. Callers choose this path explicitly after rebuilding each policy.
def _clear_policy_model_caches():
    """Rebuild policy-sensitive QE model pieces after every arithmetic-policy switch."""
    from tests.test_jqmc_kernel._generated_matrix_base import _qe_proj_pieces

    _qe_composite_pieces.cache_clear()
    _qe_proj_pieces.cache_clear()


def _assert_mixed_derivative_block(mixed_block, full_block, atol, rtol, label):
    """Compare a homogeneous block against the registered full-reference scale."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def assert_orbital_mixed_fidelity(mixed, full, label):
    """Compare finite matching blocks and prove an excess perturbation is rejected."""
    blocks = mixed_fidelity_blocks(mixed, full)
    atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
    worst = 0.0
    for mixed_block, full_block in blocks:
        _assert_mixed_derivative_block(mixed_block, full_block, atol, rtol, label)
        worst = max(worst, float(np.max(np.abs(mixed_block - full_block))))
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_mixed_derivative_block(faulty, blocks[0][1], atol, rtol, label)
    return worst


def assert_parameter_mixed_fidelity(mixed, full, label):
    """Compare finite matching blocks and prove an excess perturbation is rejected."""
    blocks = mixed_fidelity_blocks(mixed, full)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    worst = 0.0
    for mixed_block, full_block in blocks:
        _assert_mixed_derivative_block(mixed_block, full_block, atol, rtol, label)
        worst = max(worst, float(np.max(np.abs(mixed_block - full_block))))
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_mixed_derivative_block(faulty, blocks[0][1], atol, rtol, label)
    return worst

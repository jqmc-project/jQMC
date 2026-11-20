"""Canonical committed-fixture ham builders shared by the Rust equivalence suite.

Both the analytic-matrix mirror (``test_rs_analytic_matrix.py``) and
the ported per-kernel tests build their Hamiltonians here, from
COMMITTED fixtures only (``tests/trexio_example_files`` + ``tests/qe_example_files``)
-- never h5 files outside the git checkout, which a test run from the checkout cannot see.

This is a THIN adapter over the single-source builders in
``tests/test_jqmc_kernel``: ``_build_ham`` (trexio stems + the QE pgto(+)pw
composite sentinel) and ``_build_ham_for_cell`` (adds QE-projection plane-wave /
B-spline / twist-composite). We do NOT duplicate those builders -- importing keeps
one source of truth for how a committed fixture becomes a ham (radial dedup,
ragged pgto images, dense-AGP lift, active J3/DTN recipes).

``rs_ham(pattern)`` maps a short pattern name to the (stem, with_jastrow, spec, gem)
tuple the builders take. Patterns cover the (AO family x geminal x Jastrow x ECP x
BC) axes the prototype bench systems exercised, each backed by the smallest
committed fixture that flows the SAME Rust/XLA code path (a Rust==XLA equivalence
is path-sensitive, not system-sensitive, so the exact molecule is immaterial).
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

from pathlib import Path

import numpy as np

from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

# --------------------------------------------------------------------------- #
# Canonical patterns: name -> (stem, with_jastrow, spec, gem). ``spec`` selects
# the active J3 / DTN channel or the dense-AGP lift, exactly as the matrix
# builder consumes it. ``stem`` is a committed trexio file or a QE sentinel
# (``QE:*`` -- resolved by _build_ham_for_cell via _qe_proj_pieces).
# --------------------------------------------------------------------------- #
_PATTERNS: dict[str, tuple[str, bool, dict | None, str]] = {
    # ---- molecular all-electron GTO ----
    "mol_h2_cart_sd_jnone": ("H2_ae_ccpvdz_cart.h5", False, None, "SD"),
    "mol_cart_sd_jnone": ("N_ae_ccpvdz_cart.h5", False, None, "SD"),
    "mol_cart_sd_j12": ("N_ae_ccpvdz_cart.h5", True, None, "SD"),
    "mol_cart_sd_j123": ("N_ae_ccpvdz_cart.h5", True, {"j3": "cart_gto"}, "SD"),
    "mol_cart_sd_jdtn": ("N_ae_ccpvdz_cart.h5", True, {"dtn": True}, "SD"),
    "mol_cart_agp_jnone": ("H2_ae_ccpvdz_cart.h5", False, None, "AGP"),
    "mol_cart_agpn_jnone": ("H2_ae_ccpvdz_cart.h5", False, None, "AGPn"),
    "mol_sphe_sd_jnone": ("H2_ae_ccpvdz_sphe.h5", False, None, "SD"),
    "mol_sphe_agp_jnone": ("H2_ae_ccpvdz_sphe.h5", False, None, "AGP"),
    # ---- molecular ECP GTO ----
    "mol_ecp_cart_jnone": ("H2_ecp_ccpvdz_cart.h5", False, None, "SD"),
    "mol_ecp_cart_j12": ("H2_ecp_ccpvdz_cart.h5", True, None, "SD"),
    "mol_ecp_cart_j123": ("H2_ecp_ccpvdz_cart.h5", True, {"j3": "cart_gto"}, "SD"),
    "mol_ecp_cart_jdtn": ("H2_ecp_ccpvdz_cart.h5", True, {"dtn": True}, "SD"),
    "mol_ecp_sphe_jnone": ("water_ccecp_ccpvqz.h5", False, None, "SD"),
    "mol_ecp_sphe_j12": ("water_ccecp_ccpvqz.h5", True, None, "SD"),
    # ---- periodic real-Gamma ECP sphe-pGTO ----
    "pbc_gamma_sphepgto_jnone": ("lithium.h5", False, None, "SD"),
    "pbc_gamma_sphepgto_j12": ("lithium.h5", True, None, "SD"),
    "pbc_gamma_sphepgto_j123": ("lithium.h5", True, {"j3": "pgto"}, "SD"),
    "pbc_gamma_sphepgto_jdtn": ("lithium.h5", True, {"dtn": True}, "SD"),
    # ---- periodic complex twist (k != Gamma) ECP sphe-pGTO ----
    "twist_sphepgto_jnone": ("lithium_k025.h5", False, None, "SD"),
    "twist_sphepgto_j12": ("lithium_k025.h5", True, None, "SD"),
    "twist_sphepgto_j123": ("lithium_k025.h5", True, {"j3": "pgto"}, "SD"),
    "twist_sphepgto_jdtn": ("lithium_k025.h5", True, {"dtn": True}, "SD"),
    # ---- QE pgto(+)pw composite (real Gamma) ----
    "pbc_composite_jnone": ("QE:Li_gamma_pgto+pw", False, None, "SD"),
    "pbc_composite_j12": ("QE:Li_gamma_pgto+pw", True, None, "SD"),
    # main P14/P15: MO-wrapped composite J3 whose GTO child owns radial
    # exponent/coeff variational blocks.  This is intentionally distinct from
    # ``pbc_composite_j12``; without an active J3 it cannot exercise those leaves.
    "pbc_composite_j123": ("QE:Li_gamma_pgto+pw", True, {"j3": "mo"}, "SD"),
    # ---- QE plane-wave (real Gamma, ECP) ----
    "pw_gamma_jnone": ("QEPROJ:pw:Li_gamma:ECP", False, None, "SD"),
    # ---- QE B-spline (real Gamma, ECP) ----
    "bspline_gamma_jnone": ("QEPROJ:bspline:Li_gamma:ECP", False, None, "SD"),
    "pgto_bspline_gamma_jnone": ("QEPROJ:pgto+bspline:Li_gamma:ECP", False, None, "SD"),
    # ---- QE complex-twist plane-wave / B-spline (k != Gamma, ECP) ----
    "pw_twist_jnone": ("QEPROJ:pw:Li_k025:ECP", False, None, "SD"),
    "bspline_twist_jnone": ("QEPROJ:bspline:Li_k025:ECP", False, None, "SD"),
    "pgto_pw_twist_jnone": ("QEPROJ:pgto+pw:Li_k025:ECP", False, None, "SD"),
    "pgto_bspline_twist_jnone": ("QEPROJ:pgto+bspline:Li_k025:ECP", False, None, "SD"),
    # ---------------------------------------------------------------------- #
    # PRODUCTION-DEFAULT Jastrow FORM rows. The two shared
    # builders above hardcode ``jastrow_{1b,2b}_type="exp"``, but the shipped
    # default is ``pade`` (``jastrow_factor.py`` J2 L824 / the J1 twin), so
    # every row above exercises a NON-default functional form. These rows
    # re-point an existing base cell at the pade kernel branch via
    # ``_JASTROW_FORMS`` (see below); the wavefunction is otherwise identical,
    # so any divergence localises to the J1/J2 radial kernel and its
    # derivatives. Naming: ``<base>_pade`` = both channels pade (the true
    # production default); ``_p1e2`` / ``_e1p2`` = the J1 x J2 form
    # cross-product (the builders pin the two channels together, so a form
    # bug that only bites one channel would otherwise be invisible).
    # ---------------------------------------------------------------------- #
    "mol_cart_sd_j12_pade": ("N_ae_ccpvdz_cart.h5", True, None, "SD"),
    "mol_cart_sd_j12_p1e2": ("N_ae_ccpvdz_cart.h5", True, None, "SD"),
    "mol_cart_sd_j12_e1p2": ("N_ae_ccpvdz_cart.h5", True, None, "SD"),
    "mol_cart_sd_j123_pade": ("N_ae_ccpvdz_cart.h5", True, {"j3": "cart_gto"}, "SD"),
    "mol_ecp_cart_j12_pade": ("H2_ecp_ccpvdz_cart.h5", True, None, "SD"),
    "mol_ecp_sphe_j12_pade": ("water_ccecp_ccpvqz.h5", True, None, "SD"),
    "pbc_gamma_sphepgto_j12_pade": ("lithium.h5", True, None, "SD"),
    "twist_sphepgto_j12_pade": ("lithium_k025.h5", True, None, "SD"),
    "pbc_composite_j12_pade": ("QE:Li_gamma_pgto+pw", True, None, "SD"),
    # ---------------------------------------------------------------------- #
    # The QE plane-wave / B-spline Gamma leaves WITH the
    # J1+J2 Jastrow, so a driver run crosses the non-pGTO determinant leaf with
    # an active Jastrow ratio / candidate channel (the rows above are Jastrow-free).
    # ---------------------------------------------------------------------- #
    "pw_gamma_j12": ("QEPROJ:pw:Li_gamma:ECP", True, None, "SD"),
    "bspline_gamma_j12": ("QEPROJ:bspline:Li_gamma:ECP", True, None, "SD"),
    # ---------------------------------------------------------------------- #
    # The dense MO-AGP geminal (``AGP_MO``: ``AGP_data``
    # over explicit ``MOs_data`` leaves) and the three single-channel DTN masks.
    # The ``_jdtn_<mask>`` rows are the ``mol_ecp_cart_jdtn`` cell with the DTN
    # leg rebuilt under the mask in ``_DTN_MASKS`` (every other ``dtn`` row keeps
    # all of u / chi / f on).
    # ---------------------------------------------------------------------- #
    "mol_ecp_cart_agpmo_j12": ("H2_ecp_ccpvdz_cart.h5", True, None, "AGP_MO"),
    "mol_ecp_cart_jdtn_u": ("H2_ecp_ccpvdz_cart.h5", True, {"dtn": True}, "SD"),
    "mol_ecp_cart_jdtn_chi": ("H2_ecp_ccpvdz_cart.h5", True, {"dtn": True}, "SD"),
    "mol_ecp_cart_jdtn_f": ("H2_ecp_ccpvdz_cart.h5", True, {"dtn": True}, "SD"),
}

# Pattern -> ``(jastrow_1b_type, jastrow_2b_type)`` override applied AFTER the
# shared builder returns. Absent = the builders' own ``("exp", "exp")``. Both
# fields are ``pytree_node=False`` compile-time constants, so a ``.replace`` is
# the whole rewrite -- no derived state depends on the form (the kernels branch
# on the string at trace time). Rewriting here, rather than editing the shared
# 8-set builders in ``tests/test_jqmc_kernel``, keeps those single-source and
# leaves every existing exp row byte-identical.
_JASTROW_FORMS: dict[str, tuple[str, str]] = {
    "mol_cart_sd_j12_pade": ("pade", "pade"),
    "mol_cart_sd_j12_p1e2": ("pade", "exp"),
    "mol_cart_sd_j12_e1p2": ("exp", "pade"),
    "mol_cart_sd_j123_pade": ("pade", "pade"),
    "mol_ecp_cart_j12_pade": ("pade", "pade"),
    "mol_ecp_sphe_j12_pade": ("pade", "pade"),
    "pbc_gamma_sphepgto_j12_pade": ("pade", "pade"),
    "twist_sphepgto_j12_pade": ("pade", "pade"),
    "pbc_composite_j12_pade": ("pade", "pade"),
}

# A form override keyed on a pattern that does not exist would be silently
# ignored, leaving the row it was meant to create on the default ``exp`` form.
assert set(_JASTROW_FORMS) <= set(_PATTERNS), (
    f"_JASTROW_FORMS keys absent from _PATTERNS: {sorted(set(_JASTROW_FORMS) - set(_PATTERNS))}"
)


# Pattern -> ``(use_u, use_chi, use_f)`` DTN channel mask applied AFTER the shared
# builder returns. The shared builder only makes the
# all-channels-on DTN; these rows rebuild that leg with the same recipe
# (``cutoff_smoothness=4``, polynomial orders 2, ``random_init`` seed 13) and ONE
# active channel, so a driver run meets each channel's cusp constraints and
# candidate tables on its own.
_DTN_MASKS: dict[str, tuple[bool, bool, bool]] = {
    "mol_ecp_cart_jdtn_u": (True, False, False),
    "mol_ecp_cart_jdtn_chi": (False, True, False),
    "mol_ecp_cart_jdtn_f": (False, False, True),
}

assert set(_DTN_MASKS) <= set(_PATTERNS), f"_DTN_MASKS keys absent from _PATTERNS: {sorted(set(_DTN_MASKS) - set(_PATTERNS))}"


def _apply_dtn_mask(ham, mask: tuple[bool, bool, bool] | None):
    """Rebuild the DTN leg of an already-built ham with one channel mask."""
    if mask is None:
        return ham
    from jqmc.jastrow_factor import Jastrow_DTN_data

    jd = ham.wavefunction_data.jastrow_data
    if jd.jastrow_dtn_data is None:
        raise ValueError("DTN-mask override needs a ham built with an active DTN leg")
    structure_data = ham.structure_data
    coulomb_potential_data = ham.coulomb_potential_data
    atomic_numbers = np.asarray(structure_data.atomic_numbers)
    unique_atomic_numbers = np.unique(atomic_numbers)
    effective_charges = np.asarray(coulomb_potential_data._effective_charges)
    use_u, use_chi, use_f = mask
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=tuple(float(np.mean(effective_charges[atomic_numbers == Z])) for Z in unique_atomic_numbers),
        pseudized_per_species=tuple(
            bool(np.all(coulomb_potential_data.pseudized_per_atom[atomic_numbers == Z])) for Z in unique_atomic_numbers
        ),
        use_u=use_u,
        use_chi=use_chi,
        use_f=use_f,
        cutoff_smoothness=4,
        n_order_u=2,
        n_order_chi=2,
        n_order_eN=2,
        n_order_ee=2,
        random_init=True,
        seed=13,
    )
    jd = jd.replace(jastrow_dtn_data=dtn)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    ham.sanity_check()
    return ham


def _apply_jastrow_forms(ham, forms: tuple[str, str] | None):
    """Rewrite the J1 / J2 functional-form tokens on an already-built ham."""
    if forms is None:
        return ham
    j1_type, j2_type = forms
    jd = ham.wavefunction_data.jastrow_data
    j1, j2 = jd.jastrow_one_body_data, jd.jastrow_two_body_data
    if j1 is None or j2 is None:
        raise ValueError("jastrow-form override needs a ham built with_jastrow=True (J1 and J2 both present)")
    jd = jd.replace(
        jastrow_one_body_data=j1.replace(jastrow_1b_type=j1_type),
        jastrow_two_body_data=j2.replace(jastrow_2b_type=j2_type),
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


def rs_ham(pattern: str):
    """Build the committed-fixture ham for a canonical ``pattern`` name."""
    if pattern not in _PATTERNS:
        raise KeyError(f"unknown rs fixture pattern {pattern!r}; known: {sorted(_PATTERNS)}")
    stem, with_jastrow, spec, gem = _PATTERNS[pattern]
    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem)
    ham = _apply_dtn_mask(ham, _DTN_MASKS.get(pattern))
    return _apply_jastrow_forms(ham, _JASTROW_FORMS.get(pattern))


# Supplemental energy cases are deliberately separate from _PATTERNS: adding
# an energy fixture must not implicitly expand every force/parameter matrix.
_ENERGY_FIXTURE_PATTERNS = {
    "bspline_twist_jdtn": ("QEPROJ:bspline:Li_k025:ECP", True, {"dtn": True}, "SD"),
}


def rs_energy_ham(pattern: str):
    """Build the ordinary Rust fixture or an explicitly selected energy fixture."""
    if pattern in _ENERGY_FIXTURE_PATTERNS:
        from jqmc.atomic_orbital import AOs_bspline_complex_data

        stem, with_jastrow, spec, gem = _ENERGY_FIXTURE_PATTERNS[pattern]
        ham = _build_ham_for_cell(stem, with_jastrow, spec, gem)
        wf = ham.wavefunction_data
        aos = wf.geminal_data.orb_data_up_spin.aos_data
        assert isinstance(aos, AOs_bspline_complex_data)
        assert aos.k_point is not None and any(k != 0 for k in aos.k_point)
        assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
        dtn = wf.jastrow_data.jastrow_dtn_data
        assert dtn is not None and dtn.use_u and dtn.use_chi and dtn.use_f
        return ham
    return rs_ham(pattern)


def energy_with_ecp_dtn_ratio_omitted(ham, r_up, r_dn, energy):
    """Fault control: omit DTN only in ECP ratios, retaining the original kinetic term.

    Both nonlocal values use the VMC complex, twelve-point identity mesh.
    This is not a second physical reference: it constructs the specified
    faulty final energy so tests can reject it using their unchanged tolerance.
    """
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors_holomorphic

    wf = ham.wavefunction_data
    without_dtn = wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_dtn_data=None))
    cp = ham.coulomb_potential_data
    full_ecp = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(cp, wf, r_up, r_dn, np.eye(3), NN=1, Nv=12)[-1]
    no_dtn_ecp = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(cp, without_dtn, r_up, r_dn, np.eye(3), NN=1, Nv=12)[
        -1
    ]
    return energy + complex(no_dtn_ecp) - complex(full_ecp)


def build_pade_j2_dtn_no_j1_ham():
    """The one Jastrow shape the whole fixture set misses: no J1, pade J2, DTN on.

    Every ``dtn: True`` row in ``_PATTERNS`` is built with J1 present, while
    every ``_pade`` row has no DTN.  The intersection is therefore unreachable
    from ``rs_ham`` unless built explicitly.  It checks that the generic J12
    candidate/derivative tables and the independent DTN tables both advance.

    ``Jastrow_data``'s four legs are independent ``Option``s, so this shape is
    ordinary production input, not a synthetic construction.
    """
    ham = rs_ham("mol_cart_sd_jdtn")
    jd = ham.wavefunction_data.jastrow_data
    assert jd.jastrow_one_body_data is not None, "base fixture must carry a J1 to drop"
    assert jd.jastrow_dtn_data is not None, "base fixture must carry the DTN leg"
    j2 = jd.jastrow_two_body_data
    assert j2 is not None, "base fixture must carry a J2 to re-point at the pade kernel"
    jd = jd.replace(
        jastrow_one_body_data=None,
        jastrow_two_body_data=j2.replace(jastrow_2b_type="pade"),
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


def build_twist_mo_j3_ham():
    """Complex-twist fixture with a genuinely non-square MO-wrapped J3.

    The generic ``j3="mo"`` matrix fixture intentionally uses a square identity
    wrapper around its AO leaf.  That exercises the structural MO dispatch but
    cannot detect either dropping a dense complex MO projection or taking
    ``Re`` before it.  The committed ``lithium_k025.h5`` geminal instead carries
    a complex occupied-MO matrix with shape ``(1, 26)``.  Reusing that orbital as
    the phase-free J3 reference basis gives the production non-square case while
    keeping the fixture deterministic and committed-data-only. The complex MO
    coefficients stay unchanged; only the Bloch phases are removed, independently
    of the determinant's retained twist.
    """
    from jqmc.jastrow_factor import Jastrow_three_body_data, _phase_free_j3_orbitals

    ham = rs_ham("twist_sphepgto_j12")
    wf = ham.wavefunction_data
    orb = wf.geminal_data.orb_data_up_spin
    coefficients = np.asarray(orb.mo_coefficients)
    assert coefficients.ndim == 2 and coefficients.shape[0] != coefficients.shape[1]
    assert np.iscomplexobj(coefficients) and np.max(np.abs(coefficients.imag)) > 1.0e-3

    j3 = Jastrow_three_body_data.build(
        orb_data=_phase_free_j3_orbitals(orb),
        random_init=True,
        random_scale=1.0e-3,
        seed=123,
    )
    ham = ham.replace(
        wavefunction_data=wf.replace(
            jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3),
        ),
    )
    ham.sanity_check()
    return ham


def draw_cfg(ham, seed: int = 0, scale: float = 0.7):
    """Draw an ``(r_up, r_dn)`` electron configuration for ``ham`` (numpy f64)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(scale=scale, size=(nu, 3)),
        rng.normal(scale=scale, size=(nd, 3)),
    )


def draw_well_conditioned_cfg(
    ham,
    seed: int = 1,
    scale: float = 1.0,
    *,
    cond_max: float = 7.0e2,
    max_draw_tries: int = 300,
    return_metadata: bool = False,
):
    """Draw a deterministic ion-centred configuration away from a geminal node.

    The scale-invariant acceptance predicate uses only the XLA geminal value,
    before either backend derivative is evaluated.  It therefore fixes the
    numerical domain without selecting on XLA--Rust agreement or changing the
    static parity tolerance.  A determinant-magnitude cutoff is deliberately
    avoided: equivalent AGPn parameterisations can rescale ``det(G)`` by many
    orders of magnitude without changing the physical state.
    """
    from jqmc.determinant import compute_geminal_all_elements

    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for attempt in range(max_draw_tries):
        r_up = positions[np.arange(nu) % len(positions)] + rng.normal(scale=scale, size=(nu, 3))
        r_dn = positions[np.arange(nd) % len(positions)] + rng.normal(scale=scale, size=(nd, 3))
        geminal = np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))
        condition = float(np.linalg.cond(geminal))
        if np.isfinite(condition) and condition < cond_max:
            result = (np.ascontiguousarray(r_up), np.ascontiguousarray(r_dn))
            if return_metadata:
                return (*result, {"attempts": attempt + 1, "cond_g": condition})
            return result
    raise AssertionError(f"XLA could not draw cond(G) < {cond_max:g} from fixed seed={seed} after {max_draw_tries} candidates")


def build_qe_zero_k_complex_dtn_ham(kind: str, output_path: str | Path):
    """Build the v3-QE Gamma representation the twist driver actually consumes.

    Unlike ``QEPROJ:*:Li_gamma`` patterns, this goes through the production
    multi-twist converter and HDF5 loader.  The resulting Gamma orbital is a
    complex Bloch leaf at ``k=(0, 0, 0)`` and carries the shared DTN Jastrow.
    Missing committed input is a hard failure: this helper exists for blocking
    Rust/XLA contracts and must never turn an absent fixture into a skip.
    """
    from jqmc._twist_average import load_twist, load_twist_average_input
    from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_composite_data, is_complex_aos
    from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

    if kind not in {"bspline", "pgto+bspline"}:
        raise ValueError(f"unsupported zero-k complex fixture kind: {kind!r}")

    save_dir = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
    required = (save_dir / "data-file-schema.xml", save_dir / "wfc1.hdf5", save_dir / "Li.ccECP.upf")
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"committed Li_gamma QE fixture is required by this blocking Rust parity contract: {missing}"

    output_path = Path(output_path)
    convert_qe_to_twist_average_input(
        qe_save_dir=str(save_dir),
        output_path=str(output_path),
        upf_paths={"Li": str(save_dir / "Li.ccECP.upf")},
        kind=kind,
        mesh_factor=0.5,
        j1_param=None,
        j2_param=None,
        jastrow_dtn="u_chi_f",
        dtn_c=4,
        dtn_n_u=6,
        dtn_n_chi=6,
        dtn_n_eN=2,
        dtn_n_ee=2,
    )

    meta = load_twist_average_input(str(output_path))
    assert meta.n_twist == 1
    assert tuple(meta.k_points[0]) == (0.0, 0.0, 0.0)

    ham = load_twist(str(output_path), 0)
    ham.sanity_check()
    gem = ham.wavefunction_data.geminal_data
    aos = gem.orb_data_up_spin.aos_data
    child = aos.children[1] if isinstance(aos, AOs_composite_data) else aos
    assert isinstance(child, AOs_bspline_complex_data)
    assert tuple(child.k_point) == (0.0, 0.0, 0.0)
    assert is_complex_aos(aos), "zero-k v3 Gamma must dispatch through the complex production path"

    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    assert dtn is not None
    assert (dtn.use_u, dtn.use_chi, dtn.use_f) == (True, True, True)
    return ham

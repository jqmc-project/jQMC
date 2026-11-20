"""Production parity for MCMC observable aggregation, XLA vs Rust.

Scope contract:

  * The SAME walker batch is fed to both engines. The batch is built
    DETERMINISTICALLY from committed fixtures (``_rs_fixtures.rs_ham`` +
    a seeded numpy RNG) -- **no MCMC is run to produce it**, so this module is
    independent of the MCMC transition suite (sample generation).
  * ``ln|Psi|``, the local energy AND its COMPONENT breakdown (kinetic /
    bare Coulomb / Ewald ee+ei / ECP local / ECP non-local), the
    ELECTRON-WISE elements (per-electron el-ion, per-electron Ewald ei,
    per-electron grad/laplacian of ln|Det| and of the Jastrow), the
    parameter derivatives ``O_k`` and ``dE_L/dc``, the HF/Pulay forces and
    the periodic stress are pinned SEPARATELY, and only then the aggregate
    ``e_L``. A final average is never allowed to hide a cancellation: every
    component gets its own assert BEFORE the total, and the total is
    additionally closed against the sum of its own components on each engine
    (``test_mcmc_e_L_component_closure``).
  * The per-walker values are compared AS THEY STAND JUST BEFORE the driver's
    walker average / binning (``MCMC.e_L`` & friends are
    ``(step, walker, ...)`` stores; ``test_mcmc_walker_batch_*`` compares the
    per-walker vectors, and ``test_mcmc_walker_order_equivariance`` pins that
    a permuted input batch produces the identically permuted output -- a silent
    walker reorder would pass an average-only test).
  * ``test_mcmc_rust_on_off_handoff`` runs the PRODUCTION driver with the
    native backend ON and OFF on the same seed and asserts the Python
    post-processing receives the same shape, dtype, C-contiguity, key set and
    ORDER (plus values at a named tolerance) either way -- the silent dtype
    downgrade / walker reorder class of bug this item exists to catch.

XLA is the reference everywhere; a Rust-side formula is never
used to relax an oracle. Direct parity uses the dedicated XLA--Rust consistency
helper; separately documented propagated-error floors remain named constants.

Coverage: every pattern in ``_rs_fixtures._PATTERNS`` -- the production-reachable
(source x BC x twist x AE/ECP x geminal x AO-family x Jastrow) cells -- gets an
observable-parity node. Components that are not reachable for a given cell are
skipped BY CONSTRUCTION (the component table is built from the ham's own
``pbc_flag`` / ``ecp_flag`` / Jastrow leaves), never by a blanket ``skipif``.
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

import collections
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs
from jqmc._precision import precision_policy_snapshot

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_RS_PARITY_VALUE,
    MCMC_MIN_STEPS,
    RTOL_BIT_EXACT,
    RTOL_RS_PARITY_VALUE,
)
from jqmc.coulomb_potential import (
    compute_bare_coulomb_potential,
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_bare_coulomb_potential_ion_ion,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ewald_potential_ee,
    compute_ewald_potential_ei,
)
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _reduce_c_param_grads
from jqmc.determinant import (
    compute_AS_regularization_factor,
    compute_geminal_all_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_ln_det_geminal_all_elements,
)
from jqmc.hamiltonians import compute_local_energy_fast
from jqmc.jastrow_factor import (
    compute_grads_and_laplacian_Jastrow_part,
    compute_Jastrow_DTN,
    compute_Jastrow_one_body,
    compute_Jastrow_part,
    compute_Jastrow_three_body,
    compute_Jastrow_two_body,
)
from jqmc.jqmc_mcmc import (
    _NV_BASE_MESH,
    _PARAM_BLOCK_TO_MASK_KW,
    MCMC,
    _as_walker_weight,
    Nv_default,
    _geminal_inv_batched,
    _rs_native_c_param_grads,
)
from jqmc.wavefunction import compute_kinetic_energy, evaluate_ln_wavefunction
from tests._triclinic_li_ecp import TRICLINIC_CELL_MAPS, assert_cell_inverse_not_symmetric, triclinic_li_ecp_ham
from tests.test_rs_kernel._rs_fixtures import _PATTERNS, energy_with_ecp_dtn_ratio_omitted, rs_energy_ham, rs_ham
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy

pytestmark = [pytest.mark.with_chdir]

# Every production-reachable fixture cell (axis product of the fixture table), deterministic order.
_ALL_PATTERNS = sorted(_PATTERNS)
_ENERGY_PATTERNS = _ALL_PATTERNS + ["bspline_twist_jdtn"]


def _tiered(patterns):
    """``pytest.param`` rows carrying the mandatory light/heavy tier.

    Molecular committed-TREXIO cells are cheap (light); the periodic / twist /
    QE-projection cells carry image sums + big AO blocks and are heavy, matching
    how the rest of ``test_rs_kernel`` tiers the same fixtures.
    """
    rows = []
    for p in patterns:
        heavy = p.startswith(("pbc_", "twist_", "pw_", "bspline_"))
        rows.append(pytest.param(p, marks=pytest.mark.heavy if heavy else pytest.mark.light))
    return rows


_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)

# Precision zones crossed by each observable family. Backend comparisons and
# same-backend closures select their own active full/mixed central tolerance
# families below; no tolerance literal appears in this file.
_ZONES_VALUE = ("ao_eval", "mo_eval", "det_eval", "jastrow_eval")
_ZONES_KINETIC = ("wf_kinetic", "ao_grad_lap", "det_grad_lap", "jastrow_grad_lap")
_ZONES_COULOMB = ("coulomb", "ewald")
_ZONES_EL = _ZONES_VALUE + _ZONES_KINETIC + _ZONES_COULOMB + ("jastrow_ratio",)
_ZONES_GRAD = _ZONES_EL + ("mo_grad",)


# Channels that NEVER evaluate an atomic orbital. Ewald and bare
# Coulomb are functions of the electron / ion COORDINATES only; the ECP LOCAL part is a
# radial function of the same; the Jastrow VALUES are their own analytic forms. None of
# them can be affected by the pGTO far-image truncation that ``FLOOR_MEDIUM`` exists for,
# and the instrumented run confirms it: across all 26 fixture patterns (molecular,
# real-Gamma, twist, pGTO, PW, B-spline, composite) the worst deviation in this whole
# family is V_ewald_ee at abs 2.22e-14 / rel 1.30e-13 -- while the SAME run measures the
# AO-crossing channels on periodic sphe-pGTO at abs 1.37e-7 (T_kinetic), 8.28e-8
# (V_ecp_nonlocal), 4.71e-8 (ln_det). Raising this family to 2e-5 alongside them left
# NINE orders of unused headroom, through which a ~1e-7 systematic error passed
# undetected (mutation testing demonstrated exactly that).
_AO_INDEPENDENT_CHANNELS = frozenset(
    {
        "V_ewald_ee",
        "V_ewald_ei",
        "V_ewald_ei_elementwise",
        "V_bare_total",
        "V_bare_el_el",
        "V_bare_ion_ion",
        "V_bare_el_ion_elementwise",
        "V_ecp_local",
        "J1",
        "J2",
        "J3",
        "J_DTN",
    }
)


# --------------------------------------------------------------------------- #
# Deterministic fixture-built walker batch (NO MCMC involved)
# --------------------------------------------------------------------------- #
_HAM_CACHE: dict[str, object] = {}


def _ham(pattern):
    """Committed-fixture ham, cached per module run (the builders are expensive)."""
    if pattern not in _HAM_CACHE:
        _HAM_CACHE[pattern] = rs_energy_ham(pattern)
    return _HAM_CACHE[pattern]


_WALKER_DET_ABS_MIN = 1.0e-8
_WALKER_COND_MAX = 64.0
_WALKER_MAX_DRAW_TRIES = 50


def _walker_batch(ham, nw=3, seed=0, scale=0.6):
    """``(r_up_b, r_dn_b)`` of shape ``(nw, N_up/N_dn, 3)``, built deterministically.

    Candidates are scattered around the fixture's own nuclei without running an
    MCMC chain.  As in the XLA analytic-vs-autodiff tests, a fixed RNG stream is
    rejection-sampled until the XLA geminal is far enough from a node and has a
    bounded condition number.  The acceptance predicate is evaluated before either
    backend observable and never inspects XLA--Rust parity, so a failed comparison
    cannot select another walker.  Keeping the numerical domain fixed lets the
    backend tolerance remain static; it is never scaled from the observed condition.
    """
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru_batch, rd_batch = [], []
    for walker in range(nw):
        for _ in range(_WALKER_MAX_DRAW_TRIES):
            ru = np.ascontiguousarray(R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3)))
            rd = np.ascontiguousarray(R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3)))
            geminal = np.asarray(compute_geminal_all_elements(gd, ru, rd))
            det_abs = float(np.abs(np.linalg.det(geminal)))
            condition = float(np.linalg.cond(geminal))
            if (
                np.isfinite(det_abs)
                and det_abs > _WALKER_DET_ABS_MIN
                and np.isfinite(condition)
                and condition <= _WALKER_COND_MAX
            ):
                ru_batch.append(ru)
                rd_batch.append(rd)
                break
        else:
            pytest.fail(
                f"XLA could not draw walker {walker} with |det| > {_WALKER_DET_ABS_MIN:.1e} "
                f"and cond(G) <= {_WALKER_COND_MAX:g} "
                f"from fixed seed={seed} after {_WALKER_MAX_DRAW_TRIES} candidates"
            )
    return np.ascontiguousarray(ru_batch), np.ascontiguousarray(rd_batch)


def _geminal_inv(ham, ru, rd):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0])


def _is_complex(ham):
    """True for the complex-Bloch (twist k != Gamma) family.

    Uses the SAME predicate the production driver uses to set ``_wf_is_complex``
    (``atomic_orbital.is_complex_aos`` on the unwrapped AO leaf), so this module
    never diverges from the dispatch it is auditing.
    """
    from jqmc.atomic_orbital import is_complex_aos

    orb_up = ham.wavefunction_data.geminal_data.orb_data_up_spin
    ao_leaf = orb_up.aos_data if hasattr(orb_up, "aos_data") else orb_up
    return bool(is_complex_aos(ao_leaf))


def _ecp_tables(ham):
    cp = ham.coulomb_potential_data
    return (
        np.asarray(cp.structure_data.positions, float),
        np.asarray(cp._effective_charges, float),
    )


def _ewald_tables(ham):
    ed = ham.coulomb_potential_data.ewald_data
    sd = ham.structure_data
    return (
        ed,
        float(ed.eta),
        np.asarray(ed.real_image_list, float),
        np.asarray(ed.g_vectors, float),
        np.asarray(ed.g_factors, float),
        np.asarray(sd.cell, float),
        int(sd._n_max),
    )


def _cat(pair):
    return np.concatenate([np.asarray(x, float).ravel() for x in pair])


def _empty_jastrow(jd):
    """A Jastrow_data with every channel switched off (det-only reference)."""
    from jqmc.jastrow_factor import Jastrow_data

    if jd is None:
        return None
    return Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
    )


# --------------------------------------------------------------------------- #
# The component table: (name, zones, xla_thunk, rust_thunk). Built from the ham's
# OWN dispatch flags, so a cell only lists components it actually reaches.
# --------------------------------------------------------------------------- #
def _wavefunction_components(ham, ru, rd):
    """ln|Psi| pieces: ln|Det| and each present Jastrow channel, separately."""
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    cplx = _is_complex(ham)
    rows = [
        (
            "ln_det",
            _ZONES_VALUE,
            lambda: float(np.real(compute_ln_det_geminal_all_elements(gd, ru, rd))),
            (lambda: jqmc_rs.compute_ln_det_complex(gd, ru, rd))
            if cplx
            else (lambda: jqmc_rs.compute_ln_det_geminal_all_elements(gd, ru, rd)),
        )
    ]
    if jd is None:
        return rows
    if jd.jastrow_one_body_data is not None:
        j1d = jd.jastrow_one_body_data
        rows.append(
            (
                "J1",
                _ZONES_VALUE,
                lambda: float(compute_Jastrow_one_body(j1d, ru, rd)),
                lambda: jqmc_rs.compute_Jastrow_one_body(j1d, ru, rd),
            )
        )
    if jd.jastrow_two_body_data is not None:
        j2d = jd.jastrow_two_body_data
        rows.append(
            (
                "J2",
                _ZONES_VALUE,
                lambda: float(compute_Jastrow_two_body(j2d, ru, rd)),
                lambda: jqmc_rs.compute_Jastrow_two_body(j2d, ru, rd),
            )
        )
    if jd.jastrow_three_body_data is not None:
        j3d = jd.jastrow_three_body_data

        def _rust_j3():
            # The Rust J3 VALUE kernel consumes the AO block directly, so the AOs are
            # supplied from the XLA orbital API: this isolates the J3 CONTRACTION
            # (the piece the observable aggregation owns) from the AO evaluation,
            # which has its own parity nodes (``test_atomic_orbital*``).
            aos_up = np.asarray(jnp.real(j3d.compute_orb_api(j3d.orb_data, ru)), float)
            aos_dn = np.asarray(jnp.real(j3d.compute_orb_api(j3d.orb_data, rd)), float)
            return jqmc_rs.compute_Jastrow_three_body_value(np.asarray(j3d._j_matrix_jnp, float), aos_up, aos_dn)

        rows.append(("J3", _ZONES_VALUE, lambda: float(compute_Jastrow_three_body(j3d, ru, rd)), _rust_j3))
    if jd.jastrow_dtn_data is not None:
        dtn = jd.jastrow_dtn_data
        rows.append(
            (
                "J_DTN",
                _ZONES_VALUE,
                lambda: float(compute_Jastrow_DTN(dtn, ru, rd)),
                lambda: jqmc_rs.compute_dtn_value(dtn, ru, rd),
            )
        )
    return rows


def _energy_components(ham, ru, rd):
    """e_L pieces: kinetic + every Coulomb channel the cell's dispatch reaches."""
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    pbc, ecp, cplx = bool(ham.structure_data.pbc_flag), bool(cp.ecp_flag), _is_complex(ham)
    R_carts, charges = _ecp_tables(ham)
    rows = []

    if cplx:
        rows.append(
            (
                "T_kinetic_re",
                _ZONES_KINETIC,
                lambda: float(np.real(compute_kinetic_energy(wf, ru, rd))),
                lambda: jqmc_rs.compute_kinetic_withj_complex(gd, jd, ru, rd)[0],
            )
        )
        rows.append(
            (
                "T_kinetic_im",
                _ZONES_KINETIC,
                lambda: float(np.imag(compute_kinetic_energy(wf, ru, rd))),
                lambda: jqmc_rs.compute_kinetic_withj_complex(gd, jd, ru, rd)[1],
            )
        )
    else:
        rows.append(
            (
                "T_kinetic",
                _ZONES_KINETIC,
                lambda: float(compute_kinetic_energy(wf, ru, rd)),
                lambda: jqmc_rs.compute_kinetic_energy(gd, jd, ru, rd),
            )
        )

    if not pbc:
        rows.append(
            (
                "V_bare_total",
                _ZONES_COULOMB,
                lambda: float(compute_bare_coulomb_potential(cp, ru, rd)),
                lambda: jqmc_rs.compute_bare_coulomb_potential(R_carts, charges, ru, rd),
            )
        )
        rows.append(
            (
                "V_bare_el_el",
                _ZONES_COULOMB,
                lambda: float(compute_bare_coulomb_potential_el_el(ru, rd)),
                lambda: jqmc_rs.compute_bare_coulomb_potential_el_el(ru, rd),
            )
        )
        rows.append(
            (
                "V_bare_ion_ion",
                _ZONES_COULOMB,
                lambda: float(compute_bare_coulomb_potential_ion_ion(cp)),
                lambda: jqmc_rs.compute_bare_coulomb_potential_ion_ion(R_carts, charges),
            )
        )
        rows.append(
            (
                "V_bare_el_ion_elementwise",
                _ZONES_COULOMB,
                lambda: _cat(compute_bare_coulomb_potential_el_ion_element_wise(cp, ru, rd)),
                lambda: _cat(jqmc_rs.compute_bare_coulomb_potential_el_ion_element_wise(R_carts, charges, ru, rd)),
            )
        )
    else:
        ed, eta, ri, gv, gf, cell, nmax = _ewald_tables(ham)
        rows.append(
            (
                "V_ewald_ee",
                _ZONES_COULOMB,
                lambda: float(compute_ewald_potential_ee(ru, rd, ed)),
                lambda: jqmc_rs.compute_ewald_potential_ee(ru, rd, eta, ri, gv, gf, cell, nmax),
            )
        )
        rows.append(
            (
                "V_ewald_ei",
                _ZONES_COULOMB,
                lambda: float(np.sum(_cat(compute_ewald_potential_ei(cp, ru, rd, ed)))),
                lambda: jqmc_rs.compute_ewald_potential_ei(ru, rd, eta, ri, gv, gf, cell, nmax, charges, R_carts),
            )
        )
        rows.append(
            (
                "V_ewald_ei_elementwise",
                _ZONES_COULOMB,
                lambda: _cat(compute_ewald_potential_ei(cp, ru, rd, ed)),
                # the native per-electron kernel returns ONE flat (N_up + N_dn,) array
                # in the same up-then-dn order the XLA pair concatenates to.
                lambda: np.asarray(
                    jqmc_rs._ewald_ei_per_electron(ru, rd, eta, ri, gv, gf, cell, nmax, charges, R_carts, None),
                    float,
                ).ravel(),
            )
        )

    if ecp:
        rows.append(
            (
                "V_ecp_local",
                _ZONES_COULOMB,
                lambda: float(compute_ecp_local_parts_all_pairs(cp, ru, rd)),
                lambda: jqmc_rs.compute_ecp_local_parts_all_pairs(cp, ru, rd),
            )
        )
        natom = len(R_carts)
        if not pbc:
            rows.append(
                (
                    "V_ecp_nonlocal",
                    _ZONES_EL,
                    lambda: float(
                        compute_ecp_non_local_parts_all_pairs(
                            cp, wf, ru, rd, np.eye(3), Nv=Nv_default, flag_determinant_only=False
                        )[3]
                    ),
                    lambda: jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, jd, cp, ru, rd, _W, _GP, False),
                )
            )
        else:
            # PBC: XLA's all-pairs variant raises (no image awareness); the
            # nearest-neighbour variant with NN = n_atom is the matching enumeration
            # for the Rust ``nn`` argument. ``project_real=True`` (the production
            # forward-energy convention) makes the twist value real.
            rows.append(
                (
                    "V_ecp_nonlocal",
                    _ZONES_EL,
                    lambda: float(
                        np.real(
                            compute_ecp_non_local_parts_nearest_neighbors(
                                cp, wf, ru, rd, np.eye(3), NN=natom, Nv=Nv_default, flag_determinant_only=False
                            )[3]
                        )
                    ),
                    (lambda: jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, jd, cp, ru, rd, _W, _GP)[0])
                    if cplx
                    else (lambda: jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, jd, cp, ru, rd, _W, _GP, False, natom)),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# 1. Component-wise + electron-wise parity, then the aggregate (all cells)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pattern", _tiered(_ALL_PATTERNS))
def test_mcmc_lnpsi_components(pattern):
    """ln|Psi| broken into ln|Det| + J1 + J2 + J3 + J_DTN, each pinned SEPARATELY
    (Rust vs XLA) on a deterministic multi-walker batch, and only then closed
    against XLA's ``evaluate_ln_wavefunction``. A sign flip that cancels between
    two Jastrow channels cannot survive the per-channel asserts.
    """
    ham = _ham(pattern)
    wf = ham.wavefunction_data
    ru_b, rd_b = _walker_batch(ham)
    # The per-component rows (e.g. a periodic
    # J3 sum) are cancelling aggregates; mixed precision uses the reviewed medium
    # backend envelope for them, full precision keeps strict. Both closures keep
    # their strict pairs in both modes.
    component_atol, component_rtol = get_tolerance_xla_rust_consistency(
        _ZONES_VALUE, level="medium" if is_mixed_precision_enabled() else "strict"
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
    closure_atol, closure_rtol = get_tolerance_min(_ZONES_VALUE, level="strict")
    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        rows = _wavefunction_components(ham, ru, rd)
        for name, zones, xla, rust in rows:
            assert zones == _ZONES_VALUE, f"{pattern} w{w} {name}: unexpected component zones {zones}"
            ref = np.asarray(xla(), float)
            got = np.asarray(rust(), float)
            assert got.shape == ref.shape, f"{pattern} w{w} {name}: shape {got.shape} != {ref.shape}"
            np.testing.assert_allclose(
                got, ref, atol=component_atol, rtol=component_rtol, err_msg=f"{pattern} walker={w} component={name}"
            )
        # closure on the XLA side: the components ARE the whole of ln|Psi|.
        parts = float(np.real(compute_ln_det_geminal_all_elements(wf.geminal_data, ru, rd)))
        if wf.jastrow_data is not None:
            parts += float(compute_Jastrow_part(wf.jastrow_data, ru, rd))
        np.testing.assert_allclose(
            parts,
            float(np.real(evaluate_ln_wavefunction(wf, ru, rd))),
            atol=closure_atol,
            rtol=closure_rtol,
            err_msg=f"{pattern} w{w} ln|Psi| closure",
        )
        # closure on the RUST side: sum of the native components == the same ln|Psi|.
        native = float(sum(np.asarray(r[3](), float) for r in rows))
        np.testing.assert_allclose(native, parts, atol=atol, rtol=rtol, err_msg=f"{pattern} w{w} native ln|Psi| closure")


@pytest.mark.parametrize("pattern", _tiered(_ALL_PATTERNS))
def test_mcmc_local_energy_components(pattern):
    """e_L broken into kinetic + every reachable Coulomb channel (bare el-el /
    el-ion / ion-ion, Ewald ee / ei, ECP local, ECP non-local), each pinned
    SEPARATELY before the total. Includes the ELECTRON-WISE arrays (per-electron
    el-ion, per-electron Ewald ei), so a per-electron permutation or a
    compensating pair of channel errors fails here rather than cancelling in the
    scalar sum.
    """
    ham = _ham(pattern)
    ru_b, rd_b = _walker_batch(ham)
    # Each component row declares one of three reviewed zone groups; compare it
    # with exactly that group's backend pair, resolved once per group. The
    # kinetic and e_L-zone rows are cancelling
    # aggregates, so mixed precision uses the reviewed medium envelope for them;
    # full precision keeps strict. The Coulomb rows keep strict in both modes.
    atol_kinetic, rtol_kinetic = get_tolerance_xla_rust_consistency(
        _ZONES_KINETIC, level="medium" if is_mixed_precision_enabled() else "strict"
    )
    atol_coulomb, rtol_coulomb = get_tolerance_xla_rust_consistency(_ZONES_COULOMB, level="strict")
    atol_el, rtol_el = get_tolerance_xla_rust_consistency(
        _ZONES_EL, level="medium" if is_mixed_precision_enabled() else "strict"
    )
    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        for name, zones, xla, rust in _energy_components(ham, ru, rd):
            ref = np.asarray(xla(), float)
            got = np.asarray(rust(), float)
            assert got.shape == ref.shape, f"{pattern} w{w} {name}: shape {got.shape} != {ref.shape}"
            label = f"{pattern} walker={w} component={name}"
            if zones == _ZONES_KINETIC:
                np.testing.assert_allclose(got, ref, atol=atol_kinetic, rtol=rtol_kinetic, err_msg=label)
            elif zones == _ZONES_COULOMB:
                np.testing.assert_allclose(got, ref, atol=atol_coulomb, rtol=rtol_coulomb, err_msg=label)
            else:
                assert zones == _ZONES_EL, f"{label}: unexpected component zones {zones}"
                np.testing.assert_allclose(got, ref, atol=atol_el, rtol=rtol_el, err_msg=label)


@pytest.mark.parametrize("pattern", _tiered(_ALL_PATTERNS))
def test_mcmc_kinetic_electronwise(pattern):
    """The ELECTRON-WISE elements the kinetic energy is assembled from: the
    per-electron gradient / laplacian of ln|Det| and of the (J1+J2) Jastrow.
    Pinned before the contracted scalar T so a compensating up/dn error cannot
    hide in the sum. The complex-twist family has no real ln|Det| grad/lap kernel
    (its kinetic is the complex ``compute_kinetic_withj_complex``, pinned as a
    component above), so only the Jastrow rows apply there.
    """
    ham = _ham(pattern)
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    ru_b, rd_b = _walker_batch(ham)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    checked = 0
    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        if not _is_complex(ham):
            got = [np.asarray(x, float) for x in jqmc_rs.compute_grads_and_laplacian_ln_det(gd, ru, rd)]
            ref = [np.asarray(x, float) for x in compute_grads_and_laplacian_ln_Det(gd, ru, rd)]
            for g, r, tag in zip(got, ref, ("grad_up", "grad_dn", "lap_up", "lap_dn")):
                assert g.shape == r.shape, f"{pattern} w{w} ln_det {tag} shape"
                np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=f"{pattern} w{w} ln_det {tag}")
            checked += 1
        else:
            # Complex-Bloch family: the Rust kinetic is assembled INSIDE the complex
            # kernel and no per-electron grad/lap array crosses the FFI boundary, so
            # the finest available split is det-only vs det+Jastrow. Pinning BOTH
            # (against the XLA kinetic of the same wavefunction with / without the
            # Jastrow) still separates the determinant channel from the Jastrow
            # channel, so a compensating pair cannot cancel into the total.
            det_re, det_im = jqmc_rs.compute_kinetic_det_complex(gd, ru, rd)
            ref_det = complex(compute_kinetic_energy(wf.replace(jastrow_data=_empty_jastrow(jd)), ru, rd))
            np.testing.assert_allclose(
                np.array([det_re, det_im]),
                np.array([ref_det.real, ref_det.imag]),
                atol=atol,
                rtol=rtol,
                err_msg=f"{pattern} w{w} complex det-only kinetic",
            )
            withj_re, withj_im = jqmc_rs.compute_kinetic_withj_complex(gd, jd, ru, rd)
            ref_wj = complex(compute_kinetic_energy(wf, ru, rd))
            np.testing.assert_allclose(
                np.array([withj_re, withj_im]),
                np.array([ref_wj.real, ref_wj.imag]),
                atol=atol,
                rtol=rtol,
                err_msg=f"{pattern} w{w} complex det+Jastrow kinetic",
            )
            checked += 1
        # J1+J2 electron-wise grad/lap (the Rust kernel's declared scope).
        if jd is not None and jd.jastrow_three_body_data is None and jd.jastrow_dtn_data is None:
            if jd.jastrow_one_body_data is not None or jd.jastrow_two_body_data is not None:
                got = [np.asarray(x, float) for x in jqmc_rs.compute_grads_and_laplacian_jastrow_part(jd, ru, rd)]
                ref = [np.asarray(x, float) for x in compute_grads_and_laplacian_Jastrow_part(jd, ru, rd)]
                for g, r, tag in zip(got, ref, ("grad_up", "grad_dn", "lap_up", "lap_dn")):
                    assert g.shape == r.shape, f"{pattern} w{w} jastrow {tag} shape"
                    np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=f"{pattern} w{w} jastrow {tag}")
                checked += 1
    assert checked > 0, f"{pattern}: no electron-wise row exercised (audit gap, not a pass)"


# --------------------------------------------------------------------------- #
# 1b. The one input that IS legitimately shared: G^-1
# --------------------------------------------------------------------------- #
# ``ginv`` is computed once and handed to BOTH sides of
# the e_L / force / stress comparisons below. That sharing is DELIBERATE and correct
# here: it is computed by the REFERENCE implementation (``_geminal_inv_batched``, XLA),
# never by Rust, and production threads exactly one SM-maintained inverse into whichever
# backend runs -- ``compute_local_energy_fast``'s whole contract is that it consumes a
# SUPPLIED inverse. Giving each side a different G^-1 would inject a difference the
# production path does not have and would turn these rows into inverse tests.
#
# What the sharing DID hide is that Rust's own ``_matrix_inverse`` had no parity row
# anywhere in the suite (it appears only as a seed generator in test_mcmc_sweep* /
# test_single_move / test_as_regularization / test_det_streaming / test_tier2 /
# test_vmc_driver, i.e. always as an input to the thing under test, never as the thing
# under test). This row closes that gap directly.
# ``_matrix_inverse`` is the REAL kernel; the complex-Bloch (twist) cells go through the
# complex inverse instead, so they are excluded from the parametrization rather than
# skipped at runtime (a skip inside the test body would hide a future real failure).
_REAL_PATTERNS = [p for p in _ALL_PATTERNS if "twist" not in p]


@pytest.mark.parametrize("pattern", _tiered(_REAL_PATTERNS))
def test_mcmc_matrix_inverse_parity(pattern):
    """Rust ``_matrix_inverse`` == numpy's inverse, and is a true inverse of its input."""
    ham = _ham(pattern)
    assert not _is_complex(ham), f"{pattern} unexpectedly complex -- _REAL_PATTERNS filter is stale"
    gd = ham.wavefunction_data.geminal_data
    ru_b, rd_b = _walker_batch(ham)
    checked = 0
    for w in range(len(ru_b)):
        comparison_iteration = str(w)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        G = np.asarray(compute_geminal_all_elements(gd, ru, rd), dtype=np.float64)
        if G.shape[0] != G.shape[1] or G.shape[0] == 0:
            continue
        got = np.asarray(jqmc_rs._matrix_inverse(G), dtype=np.float64)
        ref = np.linalg.inv(G)
        np.testing.assert_allclose(
            got, ref, atol=ATOL_RS_PARITY_VALUE, rtol=RTOL_RS_PARITY_VALUE, err_msg=f"{pattern} w{w} _matrix_inverse"
        )
        np.testing.assert_allclose(
            G @ got,
            np.eye(G.shape[0]),
            atol=ATOL_RS_PARITY_VALUE,
            rtol=RTOL_RS_PARITY_VALUE,
            err_msg=f"{pattern} w{w} G @ G^-1 != I",
        )
        checked += 1
    assert checked > 0, f"{pattern}: no square geminal to invert (audit gap, not a pass)"


# --------------------------------------------------------------------------- #
# 2. Per-walker aggregate e_L (the value the driver stores before binning)
# --------------------------------------------------------------------------- #
def _native_e_L(ham, ru, rd, ginv):
    if _is_complex(ham):
        re, im = jqmc_rs.compute_local_energy_fast_complex(ham, ru, rd, _W, _GP, ginv)
        return complex(re, im)
    return complex(jqmc_rs.compute_local_energy_fast(ham, ru, rd, _W, _GP, ginv), 0.0)


@pytest.mark.parametrize("pattern", _tiered(_ENERGY_PATTERNS))
def test_mcmc_walker_batch_e_L(pattern):
    """PER-WALKER ``e_L`` exactly as the driver stores it before walker averaging /
    binning: a ``(num_walkers,)`` vector, native vs XLA, element by element (no
    mean is taken). Also pins the dtype/shape of the vector the Python
    post-processing receives. The supplemental committed QE Li B-spline case
    carries nonzero twist and active DTN in addition to exp J1/J2; its three
    seed-0 walkers require finite cond(G) <= 64 before energy evaluation.
    Full and mixed are compared separately against the same-mode XLA reference,
    using the existing energy tolerance and deterministic twelve-point mesh.
    """
    ham = _ham(pattern)
    ru_b, rd_b = _walker_batch(ham, seed=0)
    # Per-walker e_L is a cancelling
    # aggregate; mixed precision uses the reviewed medium backend envelope, full
    # precision keeps strict.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium" if is_mixed_precision_enabled() else "strict")
    ref, got = [], []
    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        geminal_matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, ru, rd))
        cond_g = float(np.linalg.cond(geminal_matrix))
        assert np.isfinite(cond_g) and cond_g <= _WALKER_COND_MAX
        ginv = _geminal_inv(ham, ru, rd)
        ref.append(complex(compute_local_energy_fast(ham, ru, rd, np.eye(3), ginv)))
        got.append(_native_e_L(ham, ru, rd, ginv))
    ref, got = np.asarray(ref), np.asarray(got)
    assert got.shape == ref.shape == (len(ru_b),)
    np.testing.assert_allclose(got.real, ref.real, atol=atol, rtol=rtol, err_msg=f"{pattern} per-walker Re e_L")
    np.testing.assert_allclose(got.imag, ref.imag, atol=atol, rtol=rtol, err_msg=f"{pattern} per-walker Im e_L")
    # The per-walker spread must be genuine (a constant vector would make the
    # element-wise assert vacuous).
    assert np.ptp(ref.real) > 0.0, f"{pattern}: degenerate walker batch"
    if pattern == "bspline_twist_jdtn":
        omitted = np.asarray(
            [energy_with_ecp_dtn_ratio_omitted(ham, ru, rd, energy) for ru, rd, energy in zip(ru_b, rd_b, ref, strict=True)]
        )
        assert np.all(np.isfinite(ref)) and np.all(np.isfinite(omitted))
        # The ordinary verdict compares Re and Im separately, not |e_L|.
        reference_parts = np.asarray([ref.real, ref.imag])
        omitted_parts = np.asarray([omitted.real, omitted.imag])
        assert not np.allclose(omitted_parts, reference_parts, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("pattern", _tiered(_ENERGY_PATTERNS))
def test_mcmc_e_L_component_closure(pattern):
    """The aggregate must EQUAL the sum of the components that were pinned
    individually -- on the NATIVE side. ``e_L = T + V(all channels)``: if the
    native total silently used a different channel decomposition (or dropped one),
    the components would still match XLA while the total would not close.
    The supplemental QE Li B-spline/nonzero-twist/DTN case uses three seed-0
    walkers with finite cond(G) <= 64. This closure checks the real energy;
    the companion per-walker parity test also checks its imaginary part.
    Full and mixed are checked separately with their existing same-mode policy.
    """
    ham = _ham(pattern)
    cp = ham.coulomb_potential_data
    pbc, ecp, cplx = bool(ham.structure_data.pbc_flag), bool(cp.ecp_flag), _is_complex(ham)
    ru_b, rd_b = _walker_batch(ham, seed=0)
    atol, rtol = get_tolerance_min(_ZONES_EL, level="strict")
    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        geminal_matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, ru, rd))
        cond_g = float(np.linalg.cond(geminal_matrix))
        assert np.isfinite(cond_g) and cond_g <= _WALKER_COND_MAX
        ginv = _geminal_inv(ham, ru, rd)
        rows = {r[0]: np.asarray(r[3](), float) for r in _energy_components(ham, ru, rd)}
        total = float(rows["T_kinetic_re"] if cplx else rows["T_kinetic"])
        if pbc:
            ed = cp.ewald_data
            total += float(rows["V_ewald_ee"]) + float(rows["V_ewald_ei"])
            total += float(np.real(_ewald_ii(cp, ed)))
        else:
            total += float(rows["V_bare_total"])
        if ecp:
            total += float(rows["V_ecp_local"]) + float(rows["V_ecp_nonlocal"])
        np.testing.assert_allclose(
            total,
            float(np.real(_native_e_L(ham, ru, rd, ginv))),
            atol=atol,
            rtol=rtol,
            err_msg=f"{pattern} w{w} native e_L != sum of its pinned components",
        )


def _ewald_ii(cp, ed):
    from jqmc.coulomb_potential import compute_ewald_potential_ii

    return compute_ewald_potential_ii(cp, ed)


# --------------------------------------------------------------------------- #
# 3. Walker ORDER equivariance (a silent reorder survives any average-only test)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pattern", _tiered(["mol_ecp_sphe_j12", "pbc_gamma_sphepgto_j12", "twist_sphepgto_j12"]))
def test_mcmc_walker_order_equivariance(pattern):
    """Permuting the input walker batch must permute the native per-walker
    observables IDENTICALLY (bit-for-bit -- each walker is an independent
    evaluation, so the permuted run recomputes the same fp64 expression). This is
    the check an averaged comparison structurally cannot make.
    """
    ham = _ham(pattern)
    ru_b, rd_b = _walker_batch(ham, nw=4)
    perm = np.array([2, 0, 3, 1])

    def _eval(rub, rdb):
        out = []
        for w in range(len(rub)):
            ru, rd = np.ascontiguousarray(rub[w]), np.ascontiguousarray(rdb[w])
            out.append(_native_e_L(ham, ru, rd, _geminal_inv(ham, ru, rd)))
        return np.asarray(out)

    base = _eval(ru_b, rd_b)
    permuted = _eval(ru_b[perm], rd_b[perm])
    np.testing.assert_allclose(
        permuted, base[perm], atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT, err_msg=f"{pattern} walker order"
    )


# --------------------------------------------------------------------------- #
# 4. Position derivatives (HF force / Pulay force / periodic stress), per walker
# --------------------------------------------------------------------------- #
def _native_force(ham, ru, rd, ginv):
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    pbc, ecp, cplx = bool(ham.structure_data.pbc_flag), bool(cp.ecp_flag), _is_complex(ham)
    if cplx:
        hf = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP, None)
        pp = jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd)
    elif pbc:
        hf = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv)
        pp = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv)
    elif ecp:
        hf = jqmc_rs.vmc_grad_e_L_ecp(gd, jd, cp, ru, rd, _W, _GP, ginv)
        pp = jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv)
    else:
        hf = jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, ru, rd, ginv)
        pp = jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv)
    return [np.asarray(x, float) for x in hf], [np.asarray(x, float) for x in pp]


def _xla_force(ham, ru, rd, ginv):
    RT = np.eye(3)
    ruj, rdj, gj = jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(ginv)
    hfR = np.asarray(np.real(assemble.compute_grad_e_L(ham, ruj, rdj, RT, wrt="R", geminal_inv=gj)), float)
    hfu, hfd = assemble.compute_grad_e_L(ham, ruj, rdj, RT, wrt="r", geminal_inv=gj)
    ppR = np.asarray(np.real(assemble.compute_grad_ln_Psi(ham, ruj, rdj, wrt="R", geminal_inv=gj)), float)
    ppu, ppd = assemble.compute_grad_ln_Psi(ham, ruj, rdj, wrt="r", geminal_inv=gj)
    return (
        [hfR, np.asarray(np.real(hfu), float), np.asarray(np.real(hfd), float)],
        [ppR, np.asarray(np.real(ppu), float), np.asarray(np.real(ppd), float)],
    )


# The position-derivative observables the production driver routes NATIVELY
# (jqmc_mcmc ``_rs_hf_native`` / ``_rs_pp_native``): every cell whose determinant
# leaf has a native force core. Cells whose native force is gated (complex
# MO-wrapped J3) are not covered here; MPC has no force at
# all (passive energy-only observable).
_FORCE_PATTERNS = [p for p in _ALL_PATTERNS]


@pytest.mark.parametrize("pattern", _tiered(_FORCE_PATTERNS))
def test_mcmc_walker_batch_force(pattern):
    """PER-WALKER HF force (dE_L/dR, dE_L/dr_up, dE_L/dr_dn) and Pulay force
    (dln|Psi|/dR, /dr_up, /dr_dn) -- the arrays the driver combines with the SWCT
    weights and only THEN averages. Compared walker by walker and component by
    component (dR / dr_up / dr_dn separately), never as a contracted force.
    """
    ham = _ham(pattern)
    ru_b, rd_b = _walker_batch(ham, nw=2)
    # Temporary medium envelope for the conditionally amplified mixed force
    # estimator. The same selector remains strict when all zones are fp64.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD, level="medium")
    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        ginv = _geminal_inv(ham, ru, rd)
        n_hf, n_pp = _native_force(ham, ru, rd, ginv)
        x_hf, x_pp = _xla_force(ham, ru, rd, ginv)
        for got, ref, tag in zip(n_hf + n_pp, x_hf + x_pp, ("hf_dR", "hf_dr_up", "hf_dr_dn", "pp_dR", "pp_dr_up", "pp_dr_dn")):
            assert got.shape == ref.shape, f"{pattern} w{w} {tag}: shape {got.shape} != {ref.shape}"
            assert got.dtype == ref.dtype, f"{pattern} w{w} {tag}: dtype {got.dtype} != {ref.dtype}"
            np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{pattern} w{w} {tag}")


_STRESS_PATTERNS = [p for p in _ALL_PATTERNS if p.startswith(("pbc_", "twist_", "pw_", "bspline_"))]


@pytest.mark.parametrize("pattern", _tiered(_STRESS_PATTERNS))
def test_mcmc_walker_batch_stress(pattern):
    """PER-WALKER periodic stress (HF ``dE_L/deps`` and Pulay ``dln|Psi|/deps``,
    3x3), pinned per walker before any average.

    Every production leaf is native.  In particular the pGTO+B-spline rows are
    blocking XLA-vs-Rust comparisons, so a future driver-side gate cannot make
    those cells disappear from the parity matrix.
    """
    ham = _ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    cplx = _is_complex(ham)
    ru_b, rd_b = _walker_batch(ham, nw=2)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD, level="medium")

    def _native(ru, rd, ginv):
        if cplx:
            return (
                np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP), float),
                np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd), float),
            )
        return (
            np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv), float),
            np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv), float),
        )

    for w in range(len(ru_b)):
        ru, rd = np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w])
        ginv = _geminal_inv(ham, ru, rd)
        ruj, rdj, gj = jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(ginv)
        n_hf, n_pp = _native(ru, rd, ginv)
        x_hf = np.asarray(np.real(assemble.compute_grad_e_L(ham, ruj, rdj, np.eye(3), wrt="eps", geminal_inv=gj)), float)
        x_pp = np.asarray(np.real(assemble.compute_grad_ln_Psi(ham, ruj, rdj, wrt="eps", geminal_inv=gj)), float)
        for got, ref, tag in ((n_hf, x_hf, "stress_HF"), (n_pp, x_pp, "stress_PP")):
            assert got.shape == ref.shape == (3, 3), f"{pattern} w{w} {tag} shape {got.shape}"
            np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{pattern} w{w} {tag}")


# --------------------------------------------------------------------------- #
# 5. Parameter derivatives O_k = dln|Psi|/dc and dE_L/dc, per walker
# --------------------------------------------------------------------------- #
# The production native c-axis path (``_rs_native_c_param_grads``) is gated to the
# GTO-only, real, non-basis-opt families (jqmc_mcmc ``_rs_pd_native_ok``); the
# remaining cells stay on JAX by design.
_OK_PATTERNS = [
    "mol_cart_sd_j12",
    "mol_cart_sd_j123",
    "mol_cart_sd_jdtn",
    "mol_ecp_cart_j12",
    "mol_ecp_cart_j123",
    "mol_ecp_sphe_j12",
    "mol_cart_agp_jnone",
    "mol_sphe_agp_jnone",
    "pbc_gamma_sphepgto_j12",
    # PRODUCTION-DEFAULT Jastrow form. The J1 / J2
    # c-derivative kernels (``param_grad.rs`` / ``param_grad_el.rs``) branch on
    # the SAME form token as the value/grad kernels, so the pade dln|Psi|/da and
    # dE_L/da rows are distinct code from the exp rows above.
    "mol_cart_sd_j12_pade",
    "mol_cart_sd_j12_p1e2",
    "mol_cart_sd_j12_e1p2",
    "mol_cart_sd_j123_pade",
    "mol_ecp_cart_j12_pade",
    "mol_ecp_sphe_j12_pade",
    "pbc_gamma_sphepgto_j12_pade",
]


def _c_mask():
    """The production c-axis mask: every parameter block active, basis blocks off.

    Taken verbatim from ``jqmc.derivatives.bridge._MASK_DEFAULTS`` -- the single
    source the driver itself resolves its ``opt_*`` keywords from -- so this check
    exercises the SAME family set production does, not a hand-written subset.
    """
    from jqmc.derivatives.bridge import _MASK_DEFAULTS

    return dict(_MASK_DEFAULTS)


@pytest.mark.parametrize("pattern", _tiered(_OK_PATTERNS))
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_mcmc_walker_batch_param_deriv(pattern, quantity):
    """PER-WALKER ``O_k = dln|Psi|/dc`` and ``dE_L/dc``, BLOCK BY BLOCK, native vs
    XLA on the SAME deterministic walker batch.

    The native side is the PRODUCTION assembler ``jqmc_mcmc._rs_native_c_param_grads``
    (the exact function the driver calls) and the XLA side is flattened with the
    PRODUCTION collector ``Wavefunction_data.collect_param_grads``, so the comparison
    covers the BLOCK NAMING, the per-walker stacking and the walker ORDER -- not only
    the underlying kernels. Every block is asserted individually: a per-block sign
    error that would cancel inside a contracted SR gradient fails here.
    """
    ham = _ham(pattern)
    wf = ham.wavefunction_data
    cp = ham.coulomb_potential_data
    ecp = bool(cp.ecp_flag)
    mask = _c_mask()
    ru_b, rd_b = _walker_batch(ham, nw=2)
    nw = len(ru_b)
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        cp,
        False,  # complex
        bool(ham.structure_data.pbc_flag),
        False,  # want_stress
        want_force=False,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )
    rts = np.stack([np.eye(3)] * nw)
    native = _rs_native_c_param_grads(
        ctx,
        ham,
        [np.ascontiguousarray(ru_b[w]) for w in range(nw)],
        [np.ascontiguousarray(rd_b[w]) for w in range(nw)],
        mask,
        nw,
        quantity=quantity,
        ecp_flag=ecp,
        ecp_w=_W,
        ecp_gp=_GP,
        rts_b=rts,
    )
    assert native, f"{pattern}: native c-axis assembler emitted no block"

    # XLA reference through the PRODUCTION reducer ``_reduce_c_param_grads`` (the same
    # function the flag-OFF driver calls), so the block names, the direct-MO / tie
    # handling and the (walker, ...) stacking are part of what is compared.
    per_walker = []
    for w in range(nw):
        ru, rd = jnp.asarray(ru_b[w]), jnp.asarray(rd_b[w])
        ginv = jnp.asarray(_geminal_inv(ham, np.asarray(ru), np.asarray(rd)))
        if quantity == "ln_Psi":
            per_walker.append(assemble.compute_grad_ln_Psi(ham, ru, rd, wrt="c", mask=mask, geminal_inv=ginv))
        else:
            per_walker.append(assemble.compute_grad_e_L(ham, ru, rd, np.eye(3), wrt="c", mask=mask, geminal_inv=ginv))
    grad_batched = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per_walker)
    ref_flat = _reduce_c_param_grads(wf, grad_batched, nw, mask)
    # The driver drops blocks whose per-block flag is off before storing
    # (``if not self.__param_grad_flags.get(name, True): continue``); the native
    # assembler never emits them. Apply the SAME filter to the XLA side so the
    # block-set comparison is like-for-like.
    ref_flat = {n: v for n, v in ref_flat.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}

    assert set(native) == set(ref_flat), (
        f"{pattern} {quantity}: native block set {sorted(native)} != XLA block set {sorted(ref_flat)}"
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD, level="medium")
    for block in sorted(native):
        got = np.asarray(native[block])
        ref = np.asarray(ref_flat[block])
        assert got.shape == ref.shape, f"{pattern} {block}: shape {got.shape} != {ref.shape}"
        assert got.shape[0] == nw, f"{pattern} {block}: walker axis is not axis 0 ({got.shape})"
        np.testing.assert_allclose(
            np.real(got), np.real(ref), atol=atol, rtol=rtol, err_msg=f"{pattern} {quantity} block={block} (Re)"
        )
        np.testing.assert_allclose(
            np.imag(got), np.imag(ref), atol=atol, rtol=rtol, err_msg=f"{pattern} {quantity} block={block} (Im)"
        )


# --------------------------------------------------------------------------- #
# 5b. Parameter derivatives on a NON-ORTHOGONAL (triclinic) periodic ECP cell
# --------------------------------------------------------------------------- #
# Every periodic row of ``test_mcmc_walker_batch_param_deriv`` uses the cubic committed
# lithium cell, whose cell matrix is diagonal: there ``inv(cell) == inv(cell.T)`` and a
# transposed cell inverse in the native periodic ECP parameter-derivative mesh
# (``param_grad_ecp.rs::collect_mesh``) is invisible (issue #569). These rows rebuild the
# same Li ccECP J1+J2 Gamma Hamiltonian on two triclinic cells. ``mild`` is the cell of the
# heavy strain-FD test (the committed cubic bcc cell mapped by its fixed non-symmetric
# deformation); ``skew`` additionally maps cell and nuclei by ``TRICLINIC_SKEW``
# (``tests/_triclinic_li_ecp.py``).

# Integer lattice translation (cell-vector units) of the up and of the down electrons, per
# walker. Walker 0 stays where ``_walker_batch`` put it; walkers 1 and 2 are moved whole
# lattice vectors away, so the periodic minimum-image fold of the mesh is exercised.
_TRICLINIC_WALKER_SHIFTS = (
    ((0, 0, 0), (0, 0, 0)),
    ((-1, 2, 1), (2, -1, 0)),
    ((1, -2, 2), (-2, 1, -1)),
)


def _triclinic_walker_batch(ham):
    """``_walker_batch`` (fixed seed, conditioned geminal) plus whole-lattice translations."""
    ru_b, rd_b = _walker_batch(ham, nw=len(_TRICLINIC_WALKER_SHIFTS), seed=3)
    lattice = np.asarray(ham.structure_data.cell, float)
    for w, (shift_up, shift_dn) in enumerate(_TRICLINIC_WALKER_SHIFTS):
        ru_b[w] = ru_b[w] + np.asarray(shift_up, float) @ lattice
        rd_b[w] = rd_b[w] + np.asarray(shift_dn, float) @ lattice
    return ru_b, rd_b


def _assert_triclinic_input_valid(ham, ru_b, rd_b, atol, rtol):
    """Input validity: a non-orthogonal cell whose inverse is not symmetric, live ECP and J1/J2.

    For a diagonal cell ``inv(cell).T == inv(cell)``, so a transposed cell inverse would not
    change any displacement. Here the transposed inverse differs from the inverse by more
    than the parity band itself, every cell angle differs from 90 degrees, at least one
    walker is translated by a non-zero lattice vector, and the native non-local ECP
    contribution to dE_L/d(J2) is non-zero for every walker (the periodic ECP mesh is live).
    """
    assert_cell_inverse_not_symmetric(ham, atol, rtol)
    assert any(np.any(np.asarray(s) != 0) for shifts in _TRICLINIC_WALKER_SHIFTS for s in shifts)
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    jastrow = wf.jastrow_data
    assert jastrow.jastrow_one_body_data is not None and jastrow.jastrow_two_body_data is not None
    assert ham.structure_data.pbc_flag and cp.ecp_flag
    for w in range(len(ru_b)):
        ecp_j2 = np.asarray(
            jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param(
                wf.geminal_data, jastrow, cp, np.ascontiguousarray(ru_b[w]), np.ascontiguousarray(rd_b[w]), _W, _GP
            ),
            dtype=np.float64,
        )
        assert np.all(np.isfinite(ecp_j2)) and np.any(ecp_j2 != 0.0), f"walker {w}: no live non-local ECP derivative"


@pytest.mark.heavy
@pytest.mark.parametrize("cell", sorted(TRICLINIC_CELL_MAPS))
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_mcmc_walker_batch_param_deriv_triclinic_ecp(cell, quantity):
    """Native periodic ECP ``O_k`` and ``dE_L/dc`` equal XLA on a triclinic cell.

    Same comparison as ``test_mcmc_walker_batch_param_deriv`` (production native assembler
    ``_rs_native_c_param_grads`` vs the production XLA reducer ``_reduce_c_param_grads``,
    block by block and walker by walker, J1, J2 and MO-coefficient blocks), on the Li ccECP
    J1+J2 Gamma Hamiltonian rebuilt on a non-orthogonal cell with lattice-translated walkers.
    The XLA side is the reference; full and mixed are each judged at their own policy.
    """
    ham = triclinic_li_ecp_ham(cell)
    wf = ham.wavefunction_data
    cp = ham.coulomb_potential_data
    mask = _c_mask()
    ru_b, rd_b = _triclinic_walker_batch(ham)
    nw = len(ru_b)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD, level="medium")
    _assert_triclinic_input_valid(ham, ru_b, rd_b, atol, rtol)
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        cp,
        False,  # complex
        True,  # pbc
        False,  # want_stress
        want_force=False,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )
    native = _rs_native_c_param_grads(
        ctx,
        ham,
        [np.ascontiguousarray(ru_b[w]) for w in range(nw)],
        [np.ascontiguousarray(rd_b[w]) for w in range(nw)],
        mask,
        nw,
        quantity=quantity,
        ecp_flag=True,
        ecp_w=_W,
        ecp_gp=_GP,
        rts_b=np.stack([np.eye(3)] * nw),
    )
    per_walker = []
    for w in range(nw):
        ru, rd = jnp.asarray(ru_b[w]), jnp.asarray(rd_b[w])
        ginv = jnp.asarray(_geminal_inv(ham, np.asarray(ru), np.asarray(rd)))
        if quantity == "ln_Psi":
            per_walker.append(assemble.compute_grad_ln_Psi(ham, ru, rd, wrt="c", mask=mask, geminal_inv=ginv))
        else:
            per_walker.append(assemble.compute_grad_e_L(ham, ru, rd, np.eye(3), wrt="c", mask=mask, geminal_inv=ginv))
    grad_batched = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per_walker)
    ref_flat = _reduce_c_param_grads(wf, grad_batched, nw, mask)
    ref_flat = {n: v for n, v in ref_flat.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}

    # The J1, J2 and MO-coefficient blocks must all be present and non-vacuous on both sides.
    assert set(native) == set(ref_flat), f"{cell} {quantity}: native blocks {sorted(native)} != XLA {sorted(ref_flat)}"
    assert {"j1_param", "j2_param", "mo_coefficients_up"} <= set(native), sorted(native)
    for block in sorted(native):
        got = np.asarray(native[block])
        ref = np.asarray(ref_flat[block])
        assert got.shape == ref.shape, f"{cell} {block}: shape {got.shape} != {ref.shape}"
        assert got.shape[0] == nw, f"{cell} {block}: walker axis is not axis 0 ({got.shape})"
        assert np.all(np.isfinite(got)) and np.any(ref != 0.0), f"{cell} {quantity} block={block}: vacuous reference"
        np.testing.assert_allclose(
            np.real(got), np.real(ref), atol=atol, rtol=rtol, err_msg=f"triclinic {cell} {quantity} block={block} (Re)"
        )
        np.testing.assert_allclose(
            np.imag(got), np.imag(ref), atol=atol, rtol=rtol, err_msg=f"triclinic {cell} {quantity} block={block} (Im)"
        )


# --------------------------------------------------------------------------- #
# 6. Rust ON / OFF hand-off: same Python post-processing, same shape/dtype/order
# --------------------------------------------------------------------------- #
_STORE_NAMES = (
    "w_L",
    "e_L",
    "e_L2",
    "force_HF_stored",
    "force_PP_stored",
    "E_L_force_PP_stored",
    "stress_HF_stored",
    "stress_PP_stored",
    "E_L_stress_PP_stored",
)
# Stores carrying the local energy's dtype (complex128 on a complex twist, float64 otherwise).
_E_L_TYPED_STORE_NAMES = frozenset({"e_L", "E_L_force_PP_stored", "E_L_stress_PP_stored"})


def _run_driver(
    ham,
    rust_on,
    seed=34356,
    nw=2,
    *,
    comput_position_deriv=True,
    comput_stress=True,
    comput_param_deriv=True,
    native_counts=None,
):
    """One PRODUCTION ``MCMC`` run with the native backend forced on/off.

    Rust and XLA use different production RNG transforms. This test checks the
    SHAPE / DTYPE / ORDER / KEY-SET hand-off contract. Deterministic numerical
    parity is covered in ``test_chain_modes.py``. ``native_counts`` (a Counter),
    when given, receives the per-method native obs-ctx call counts of the run.
    """
    env_flag = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=seed,
            epsilon_AS=1.0e-6,
            num_walkers=nw,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
            comput_log_WF_param_deriv=comput_param_deriv,
            comput_e_L_param_deriv=comput_param_deriv,
            random_discretized_mesh=True,
        )
        with native_ctx_spy() as counts:
            mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
        # Assert the BACKEND, do not assume it. An ON side built purely by popping
        # JQMC_FLAG_RUST_ON_CPU and trusting the default would, if the default flipped or a
        # gate silently routed this family back to XLA, make both sides of the hand-off
        # comparison the same JAX run, and every assertion below (shape / dtype /
        # contiguity / values) would pass vacuously.
        if rust_on:
            assert_native_ran(counts, "MCMC observables rust-ON handoff run")
        else:
            assert_no_native_ran(counts, "MCMC observables rust-OFF handoff run")
        if native_counts is not None:
            native_counts.update(counts)
        stores = {n: getattr(mcmc, n) for n in _STORE_NAMES}
        stores["__dln_Psi_dc"] = mcmc.dln_Psi_dc
        stores["__de_L_dc"] = mcmc.de_L_dc
        return stores
    finally:
        if env_flag is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = env_flag


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern,comput_position_deriv,comput_stress,comput_param_deriv",
    [
        pytest.param("mol_cart_sd_j12", True, True, True, id="mol-force"),
        pytest.param("mol_ecp_cart_j12", True, True, True, id="mol-ecp-force"),
        pytest.param("pbc_gamma_sphepgto_j12", True, False, False, id="pbc-force-only"),
        pytest.param("pbc_gamma_sphepgto_j12", False, True, False, id="pbc-stress-only"),
        # Complex twist, stress requested, atomic force off. The early
        # complex-chunk JAX-init skip must read the stress request as well as
        # comput_position_deriv, or the run raises "diverged from the authoritative
        # native chunk gate" at run start.
        pytest.param("twist_sphepgto_j12", False, True, False, id="twist-stress-only"),
    ],
)
def test_mcmc_rust_on_off_handoff(pattern, comput_position_deriv, comput_stress, comput_param_deriv):
    """The SAME Python post-processing must receive the SAME shape, dtype,
    C-contiguity, key set (in the SAME order) and per-walker values whether the
    native backend is ON or OFF. A silent fp64->fp32 downgrade, a transposed
    ``(walker, step)`` store, or a walker reorder inside the native hand-off all
    fail here even though the binned mean would be unchanged.
    """
    ham = _ham(pattern)
    off = _run_driver(
        ham,
        rust_on=False,
        comput_position_deriv=comput_position_deriv,
        comput_stress=comput_stress,
        comput_param_deriv=comput_param_deriv,
    )
    on_counts = collections.Counter()
    on = _run_driver(
        ham,
        rust_on=True,
        comput_position_deriv=comput_position_deriv,
        comput_stress=comput_stress,
        comput_param_deriv=comput_param_deriv,
        native_counts=on_counts,
    )
    if _is_complex(ham) and comput_stress:
        # Complex-twist native chunks are energy-only, so a stress request must take the
        # per-step dispatch and call the native complex stress kernels once per step.
        assert on_counts["run_energy_chunk_batch"] == 0, f"{pattern}: stress run took the chunk {dict(on_counts)}"
        for kernel in ("grad_e_L_eps_pbc_complex_batch", "grad_ln_psi_eps_pbc_complex_batch"):
            assert on_counts[kernel] == MCMC_MIN_STEPS, f"{pattern}: {kernel} calls {dict(on_counts)}"

    assert list(on.keys()) == list(off.keys())
    for name in _STORE_NAMES:
        a, b = np.asarray(on[name]), np.asarray(off[name])
        assert a.shape == b.shape, f"{pattern} {name}: shape {a.shape} (rust) != {b.shape} (jax)"
        assert a.dtype == b.dtype, f"{pattern} {name}: dtype {a.dtype} (rust) != {b.dtype} (jax)"
        # Complex-twist runs keep Im(e_L): the e_L-typed stores are complex128 there.
        expected = np.complex128 if (_is_complex(ham) and name in _E_L_TYPED_STORE_NAMES) else np.float64
        assert a.dtype == expected, f"{pattern} {name}: dtype {a.dtype}, expected {np.dtype(expected)}"
        assert a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"], f"{pattern} {name}: contiguity differs"
        # (step, walker, ...) ordering: axis 0 is the chain, axis 1 the walkers.
        assert a.ndim >= 2 and a.shape[1] == 2, f"{pattern} {name}: walker axis is not axis 1 ({a.shape})"
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{pattern} {name}: non-finite values"

    for key in ("__dln_Psi_dc", "__de_L_dc"):
        da, db = on[key], off[key]
        assert list(da.keys()) == list(db.keys()), f"{pattern} {key}: block ORDER/set differs {list(da)} vs {list(db)}"
        for block in da:
            a, b = np.asarray(da[block]), np.asarray(db[block])
            assert a.shape == b.shape, f"{pattern} {key}[{block}]: shape {a.shape} != {b.shape}"
            assert a.dtype == b.dtype, f"{pattern} {key}[{block}]: dtype {a.dtype} != {b.dtype}"
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{pattern} {key}[{block}]: non-finite values"


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern,epsilon,steps,walkers",
    [
        pytest.param("mol_ecp_sphe_j12", 0.1, 60, 3, id="real-default-epsilon"),
        pytest.param("twist_sphepgto_j12", 2.0, 3, 2, id="complex-active-epsilon"),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_native_chunk_stored_as_weights_match_recomputed_walkers(pattern, epsilon, steps, walkers, monkeypatch):
    """Stored native weights use the squared AS ratio at the returned walkers.

    Water uses the API-default epsilon, with a fixed seed that reaches the
    regularized branch. The complex Li determinant is 1x1, hence R_AS=1;
    epsilon=2 makes its weight 1/4 rather than the vacuous unit weight.
    Neither fixture is screened by its backend error. Positions are generated
    by the real native chunk, not an injected trajectory. An independent NumPy
    inverse/norm reference anchors the XLA factor and the closed-form weight.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham = rs_ham(pattern)
    geminal = ham.wavefunction_data.geminal_data
    driver = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=epsilon,
        num_walkers=walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    atol, rtol = get_tolerance_xla_rust_consistency(("ao_eval", "det_eval", "det_ratio"), "medium")
    compared = 0
    active = 0
    with native_ctx_spy() as counts:
        for _ in range(steps):
            driver.run(num_mcmc_steps=1)
            up = np.asarray(driver._MCMC__latest_r_up_carts)
            dn = np.asarray(driver._MCMC__latest_r_dn_carts)
            stored = np.asarray(driver.w_L)[-1]
            assert stored.shape == (walkers,) and np.isfinite(stored).all()
            for walker in range(walkers):
                matrix = np.asarray(compute_geminal_all_elements(geminal, up[walker], dn[walker]))
                inverse = np.linalg.inv(matrix)
                norm = min(np.sum(np.abs(matrix) ** 2, axis=1).min(), np.sum(np.abs(matrix) ** 2, axis=0).min())
                factor = (norm * np.sum(np.abs(inverse) ** 2)) ** (-3.0 / 8.0)
                reference = (factor / max(factor, epsilon)) ** 2
                xla_factor = compute_AS_regularization_factor(geminal, up[walker], dn[walker])
                xla_weight = np.asarray(_as_walker_weight(xla_factor, epsilon))
                assert np.isfinite(factor) and factor > 0 and np.isfinite(reference)
                np.testing.assert_allclose(xla_weight, reference, atol=atol, rtol=rtol)
                np.testing.assert_allclose(stored[walker], xla_weight, atol=atol, rtol=rtol)
                np.testing.assert_allclose(stored[walker], reference, atol=atol, rtol=rtol)
                active += int(factor < epsilon)
                compared += 1
    assert counts["run_energy_chunk_batch"] == steps, dict(counts)
    assert compared == steps * walkers
    assert active > 0, "no compared walker reached R_AS < epsilon; unit weights cannot detect a wrong power"

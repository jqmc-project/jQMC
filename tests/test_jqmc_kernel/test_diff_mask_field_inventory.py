"""Completeness gate for the parameter-gradient mask.

``jqmc._diff_mask.apply_diff_mask`` decides which leaves of a disabled block are
frozen from ``diff_tag`` field metadata or, failing that, from the field NAME
(``_PARAM_FIELD_NAMES`` / ``_COORD_FIELD_NAMES``); every other field is left
differentiable.  ``Wavefunction_data.with_param_grad_mask`` freezes the J1 / J2 /
J3 / geminal blocks through that name route and the DTN channels and GTO basis
arrays through explicit per-leaf code.  Nothing tied the name list to the
dataclasses it walks: a renamed or newly added variational array would silently
stay differentiable while the user asked for it to be held fixed.

This module closes that with three checks.

1. Inventory: every pytree (``pytree_node=True``) leaf field of every flax
   dataclass reachable from ``Wavefunction_data`` (walked through the type hints,
   including every dataclass sibling under a shared jqmc base class, so AGPn / SD
   and every AO leaf family are included) must appear in the reviewed table
   ``_REVIEWED_FIELDS`` with a role and a reason, and the table must have no
   stale entries.  Adding or renaming a field fails here until it is reviewed.
2. Static classification: a field reviewed as a name-masked variational
   parameter must be classified ``"param"`` by the same rule ``apply_diff_mask``
   uses (``diff_tag`` metadata, else ``_PARAM_FIELD_NAMES``); the coordinate
   field must be classified ``"coord"``.
3. Behaviour, with "variational" defined by what the optimizer differentiates:
   on each fixture every block returned by ``get_variational_blocks`` (all
   flags on) must map to reviewed variational source fields whose values equal
   the block values; its gradient must be nonzero with every flag on and
   exactly zero with only its own flag off, while every block of another flag
   still flows.

Negative controls: a synthetic dataclass with an unreviewed field fails the
inventory with the specific message, and removing ``jastrow_1b_param`` from
``_PARAM_FIELD_NAMES`` (a red-making rename) fails both the static
classification and the behavioural check.  The verdicts are exact (zero versus
nonzero), so the same checks apply in full and mixed precision.
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
import os
import sys
import typing
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import struct

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jqmc._diff_mask as diff_mask_module
from jqmc.determinant import Geminal_data, to_AGP
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _phase_free_j3_orbitals,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests._ecp_core_electrons import core_electrons_of

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_STATE_TOKEN = ("diff-mask-inventory", 0)

# ---------------------------------------------------------------------------
# Reviewed table: every pytree leaf field reachable from Wavefunction_data.
# Roles:
#   "param-name"     variational; frozen by the whole-block apply_diff_mask route,
#                    so it must be classified "param" by diff_tag / name.
#   "param-surgical" variational; frozen by explicit per-leaf stop_gradient code
#                    in with_param_grad_mask (DTN channels, GTO radial arrays).
#   "coord"          coordinate leaf; must be classified "coord".
#   "fixed"          not a variational parameter (never a get_variational_blocks
#                    source); the reason says why.
# ---------------------------------------------------------------------------
_REVIEWED_FIELDS: dict[tuple[str, str], tuple[str, str]] = {
    ("AGP_data", "lambda_matrix"): ("param-name", "AGP geminal matrix, block lambda_matrix (opt_geminal)"),
    ("AGPn_data", "lambda_diag"): ("param-name", "AGPn occupation diagonal, block lambda_diag (opt_geminal)"),
    ("AGPn_data", "lambda_unpaired"): ("param-name", "AGPn open-shell block, block lambda_unpaired (opt_geminal)"),
    ("MOs_data", "mo_coefficients"): (
        "param-name",
        "blocks mo_coefficients_up/_dn for SD/AGPn (opt_geminal); frozen storage for AGP and J3 orbitals",
    ),
    ("AO_radial_shell_data", "exponents"): (
        "param-surgical",
        "GTO basis exponents, blocks j3_basis_exp / lambda_basis_exp (opt_*_basis_exp)",
    ),
    ("AO_radial_shell_data", "coefficients"): (
        "param-surgical",
        "GTO contraction coefficients, blocks j3_basis_coeff / lambda_basis_coeff (opt_*_basis_coeff)",
    ),
    ("Jastrow_one_body_data", "jastrow_1b_param"): ("param-name", "J1 parameter, block j1_param (opt_J1_param)"),
    ("Jastrow_two_body_data", "jastrow_2b_param"): ("param-name", "J2 parameter, block j2_param (opt_J2_param)"),
    ("Jastrow_three_body_data", "j_matrix"): ("param-name", "J3 matrix, block j3_matrix (opt_J3_param)"),
    ("Jastrow_DTN_data", "alpha_indep"): ("param-surgical", "DTN u channel, block j3_poly_alpha (opt_JDTN_u_param)"),
    ("Jastrow_DTN_data", "beta_indep"): ("param-surgical", "DTN chi channel, block j3_poly_beta (opt_JDTN_chi_param)"),
    ("Jastrow_DTN_data", "gamma_indep"): ("param-surgical", "DTN f channel, block j3_poly_gamma (opt_JDTN_f_param)"),
    ("Jastrow_DTN_data", "cutoff_scale"): (
        "fixed",
        "comoving strain scale; stored as 1.0 and set only by _apply_strain, never an optimizer block",
    ),
    ("Structure_data", "positions"): ("coord", "nuclear positions; flow for forces, masked only by DiffMask.coords"),
    ("Structure_data", "vec_a"): ("fixed", "lattice vector; strain enters through _apply_strain, not a block"),
    ("Structure_data", "vec_b"): ("fixed", "lattice vector; strain enters through _apply_strain, not a block"),
    ("Structure_data", "vec_c"): ("fixed", "lattice vector; strain enters through _apply_strain, not a block"),
    ("AOs_bspline_data", "spline_coefficients"): (
        "fixed",
        "B-spline control points; basis optimization is not exposed for spline leaves (no block)",
    ),
    ("AOs_bspline_complex_data", "spline_coefficients"): (
        "fixed",
        "complex B-spline control points; basis optimization is not exposed for spline leaves (no block)",
    ),
    ("AOs_bspline_tiled_complex_data", "spline_coefficients"): (
        "fixed",
        "provisional tiled spline control points; no basis optimization block",
    ),
    ("OrbitalKBlock_data", "coefficients"): (
        "fixed",
        "frozen local tiled operator; SD rejects orbital optimization and contraction stops its gradient",
    ),
    ("AOs_plane_wave_data", "phase_is_sin"): ("fixed", "constant cos/sin selector of the real plane-wave basis"),
    ("AOs_cart_pgto_complex_data", "_image_phase_factors"): (
        "fixed",
        "Bloch image phases exp(i k.L), derived from k_point and the lattice",
    ),
    ("AOs_sphe_pgto_complex_data", "_image_phase_factors"): (
        "fixed",
        "Bloch image phases exp(i k.L), derived from k_point and the lattice",
    ),
    ("_Tiled_cart_pgto_complex_view", "_image_phase_factors"): (
        "fixed",
        "transient primitive Bloch phases; no variational block",
    ),
    ("_Tiled_sphe_pgto_complex_view", "_image_phase_factors"): (
        "fixed",
        "transient primitive Bloch phases; no variational block",
    ),
    ("_Tiled_cart_pgto_complex_view", "center_offsets"): (
        "fixed",
        "transient displacement derived from physical coordinates during evaluation, never an optimizer block",
    ),
    ("_Tiled_sphe_pgto_complex_view", "center_offsets"): (
        "fixed",
        "transient displacement derived from physical coordinates during evaluation, never an optimizer block",
    ),
}

_VARIATIONAL_ROLES = ("param-name", "param-surgical")

_ALL_FLAGS = (
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
_ALL_ON = dict.fromkeys(_ALL_FLAGS, True)


# ---------------------------------------------------------------------------
# Inventory walk
# ---------------------------------------------------------------------------
def _subclasses(cls: type) -> set[type]:
    out = {cls}
    for sub in cls.__subclasses__():
        out |= _subclasses(sub)
    return out


def _family(cls: type) -> set[type]:
    """``cls`` plus every dataclass under any jqmc base class of ``cls``.

    ``Wavefunction_data.geminal_data`` is annotated with the ``AGP_data`` alias
    while production also stores ``AGPn_data`` / ``SD_data`` (siblings under
    ``_GeminalBase``); AO annotations name one GTO class while every AO leaf
    family shares ``AOs_base_data``.  Walking the jqmc bases covers them.
    """
    members = {cls} | _subclasses(cls)
    for base in cls.__mro__:
        if base is object or not base.__module__.startswith("jqmc"):
            continue
        members |= _subclasses(base)
    return {c for c in members if dataclasses.is_dataclass(c)}


def _dataclass_types(hint: object) -> set[type]:
    if typing.get_origin(hint) is not None:
        found: set[type] = set()
        for arg in typing.get_args(hint):
            found |= _dataclass_types(arg)
        return found
    # Abstract bases such as ``AOs_base_data`` are not dataclasses themselves;
    # their dataclass subclasses are what a field can hold.
    if isinstance(hint, type) and (dataclasses.is_dataclass(hint) or hint.__module__.startswith("jqmc")):
        return _family(hint)
    return set()


def pytree_leaf_fields(root: type) -> set[tuple[str, str]]:
    """Every ``(class, field)`` pytree field that is not itself a dataclass container."""
    seen: set[type] = set()
    leaves: set[tuple[str, str]] = set()
    stack = [root]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if not f.metadata.get("pytree_node", True):
                continue
            children = _dataclass_types(hints[f.name])
            if children:
                stack.extend(children)
            else:
                leaves.add((cls.__qualname__, f.name))
    return leaves


def check_inventory(root: type, reviewed: dict[tuple[str, str], tuple[str, str]]) -> None:
    found = pytree_leaf_fields(root)
    unreviewed = sorted(found - set(reviewed))
    stale = sorted(set(reviewed) - found)
    assert not unreviewed, (
        "unreviewed pytree leaf field(s) reachable from "
        f"{root.__qualname__}: {', '.join(f'{c}.{n}' for c, n in unreviewed)}; classify each in _REVIEWED_FIELDS "
        "(variational fields must also be frozen by with_param_grad_mask)"
    )
    assert not stale, f"stale _REVIEWED_FIELDS entries (field no longer exists): {stale}"


def _classes_by_name(root: type) -> dict[str, type]:
    seen: dict[str, type] = {}
    stack = [root]
    while stack:
        cls = stack.pop()
        if cls.__qualname__ in seen:
            continue
        seen[cls.__qualname__] = cls
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if f.metadata.get("pytree_node", True):
                stack.extend(_dataclass_types(hints[f.name]))
    return seen


def static_diff_tag(f: dataclasses.Field) -> str | None:
    """The classification rule of ``apply_diff_mask`` (read at call time)."""
    if f.metadata and "diff_tag" in f.metadata:
        return f.metadata["diff_tag"]
    if f.name in diff_mask_module._PARAM_FIELD_NAMES:
        return "param"
    if f.name in diff_mask_module._COORD_FIELD_NAMES:
        return "coord"
    return None


def check_static_classification(root: type, reviewed: dict[tuple[str, str], tuple[str, str]]) -> None:
    classes = _classes_by_name(root)
    wrong = []
    for (cls_name, field_name), (role, _reason) in sorted(reviewed.items()):
        fld = {f.name: f for f in dataclasses.fields(classes[cls_name])}[field_name]
        tag = static_diff_tag(fld)
        expected = {"param-name": "param", "coord": "coord"}.get(role)
        if expected is not None and tag != expected:
            wrong.append(
                f"{cls_name}.{field_name} (role {role}) is classified {tag!r} by apply_diff_mask, expected {expected!r}"
            )
    assert not wrong, "diff-mask classification mismatch: " + "; ".join(wrong)


# ---------------------------------------------------------------------------
# Behavioural check
# ---------------------------------------------------------------------------
def _aos(orb):
    return orb.aos_data if hasattr(orb, "mo_coefficients") else orb


# block name -> (controlling flag, reviewed source (class, field), getters of the source leaves)
def _block_sources(name: str):
    jd = lambda w: w.jastrow_data  # noqa: E731
    gem = lambda w: w.geminal_data  # noqa: E731
    table = {
        "j1_param": (
            "opt_J1_param",
            ("Jastrow_one_body_data", "jastrow_1b_param"),
            [lambda w: jd(w).jastrow_one_body_data.jastrow_1b_param],
        ),
        "j2_param": (
            "opt_J2_param",
            ("Jastrow_two_body_data", "jastrow_2b_param"),
            [lambda w: jd(w).jastrow_two_body_data.jastrow_2b_param],
        ),
        "j3_matrix": (
            "opt_J3_param",
            ("Jastrow_three_body_data", "j_matrix"),
            [lambda w: jd(w).jastrow_three_body_data.j_matrix],
        ),
        "j3_poly_alpha": (
            "opt_JDTN_u_param",
            ("Jastrow_DTN_data", "alpha_indep"),
            [lambda w: jd(w).jastrow_dtn_data.alpha_indep],
        ),
        "j3_poly_beta": (
            "opt_JDTN_chi_param",
            ("Jastrow_DTN_data", "beta_indep"),
            [lambda w: jd(w).jastrow_dtn_data.beta_indep],
        ),
        "j3_poly_gamma": (
            "opt_JDTN_f_param",
            ("Jastrow_DTN_data", "gamma_indep"),
            [lambda w: jd(w).jastrow_dtn_data.gamma_indep],
        ),
        "lambda_matrix": ("opt_geminal", ("AGP_data", "lambda_matrix"), [lambda w: gem(w).lambda_matrix]),
        "lambda_diag": ("opt_geminal", ("AGPn_data", "lambda_diag"), [lambda w: gem(w).lambda_diag]),
        "lambda_unpaired": ("opt_geminal", ("AGPn_data", "lambda_unpaired"), [lambda w: gem(w).lambda_unpaired]),
        "mo_coefficients_up": (
            "opt_geminal",
            ("MOs_data", "mo_coefficients"),
            [lambda w: gem(w).orb_data_up_spin.mo_coefficients],
        ),
        "mo_coefficients_dn": (
            "opt_geminal",
            ("MOs_data", "mo_coefficients"),
            [lambda w: gem(w).orb_data_dn_spin.mo_coefficients],
        ),
        "j3_basis_exp": (
            "opt_J3_basis_exp",
            ("AO_radial_shell_data", "exponents"),
            [lambda w: _aos(jd(w).jastrow_three_body_data.orb_data).radial.exponents],
        ),
        "j3_basis_coeff": (
            "opt_J3_basis_coeff",
            ("AO_radial_shell_data", "coefficients"),
            [lambda w: _aos(jd(w).jastrow_three_body_data.orb_data).radial.coefficients],
        ),
        "lambda_basis_exp": (
            "opt_lambda_basis_exp",
            ("AO_radial_shell_data", "exponents"),
            [
                lambda w: _aos(gem(w).orb_data_up_spin).radial.exponents,
                lambda w: _aos(gem(w).orb_data_dn_spin).radial.exponents,
            ],
        ),
        "lambda_basis_coeff": (
            "opt_lambda_basis_coeff",
            ("AO_radial_shell_data", "coefficients"),
            [
                lambda w: _aos(gem(w).orb_data_up_spin).radial.coefficients,
                lambda w: _aos(gem(w).orb_data_dn_spin).radial.coefficients,
            ],
        ),
    }
    return table.get(name)


def _grad(wf, r_up, r_dn, flags):
    def f(w):
        return evaluate_ln_wavefunction(w.with_param_grad_mask(**flags), r_up, r_dn)

    return jax.grad(f)(wf)


def _maxabs(getters, tree) -> float:
    return max(float(np.max(np.abs(np.asarray(g(tree))))) for g in getters)


def check_flag_freezes_its_blocks(wf, r_up, r_dn, reviewed) -> dict[str, str]:
    """Return ``{block: flag}`` after asserting the per-flag freeze semantics."""
    blocks = wf.get_variational_blocks(state_token=_STATE_TOKEN, **_ALL_ON)
    assert blocks, "fixture exposes no variational block"
    sources = {}
    for blk in blocks:
        entry = _block_sources(blk.name)
        assert entry is not None, f"variational block {blk.name!r} has no reviewed source field in this gate"
        flag, field_key, getters = entry
        role = reviewed.get(field_key, ("<missing>", ""))[0]
        assert role in _VARIATIONAL_ROLES, f"block {blk.name!r} is sourced from {field_key}, reviewed as {role!r}"
        expected = np.concatenate([np.ravel(np.asarray(g(wf))) for g in getters])
        np.testing.assert_array_equal(np.ravel(np.asarray(blk.values)), expected, err_msg=f"block {blk.name!r} source")
        sources[blk.name] = (flag, getters)

    g_on = _grad(wf, r_up, r_dn, _ALL_ON)
    for name, (_flag, getters) in sources.items():
        assert _maxabs(getters, g_on) > 0.0, f"block {name!r} carries no gradient with every flag on (vacuous)"

    for flag in sorted({flag for flag, _ in sources.values()}):
        g_off = _grad(wf, r_up, r_dn, {**_ALL_ON, flag: False})
        for name, (owner, getters) in sources.items():
            leak = _maxabs(getters, g_off)
            if owner == flag:
                assert leak == 0.0, f"block {name!r} is not frozen with {flag}=False (max |grad| = {leak:.3e})"
            else:
                assert leak > 0.0, f"block {name!r} ({owner}) was frozen by {flag}=False"
    return {name: flag for name, (flag, _) in sources.items()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _positions(structure, n_up, n_dn, seed):
    rng = np.random.RandomState(seed)
    centers = np.asarray(structure._positions_cart_np)

    def pick(n):
        base = centers[rng.randint(0, centers.shape[0], size=n)]
        return jax.numpy.asarray(base + 0.4 * rng.standard_normal((n, 3)))

    return pick(n_up), pick(n_dn)


def _load(name):
    return read_trexio_file(trexio_file=os.path.join(TREXIO_DIR, name), store_tuple=True)


def _jastrow(sd, aos, cpd, *, j3_orb=None, dtn=False):
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=sd,
        core_electrons=core_electrons_of(cpd, len(sd.atomic_numbers)),
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=_phase_free_j3_orbitals(aos if j3_orb is None else j3_orb),
        random_init=True,
        random_scale=0.05,
        seed=1,
    )
    dtn_data = None
    if dtn:
        dtn_data = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            use_u=True,
            use_chi=True,
            use_f=True,
            cutoff_smoothness=4,
            n_order_u=4,
            n_order_chi=4,
            n_order_eN=2,
            n_order_ee=2,
            L_u=5.0,
            L_chi=5.0,
            L_f=5.0,
            random_init=True,
            seed=2,
        )
    return Jastrow_data(
        jastrow_one_body_data=j1,
        jastrow_two_body_data=j2,
        jastrow_three_body_data=j3,
        jastrow_dtn_data=dtn_data,
    )


def _fx_agp_mo_dtn():
    sd, aos, _mu, _md, gem, cpd = _load("H2_ae_ccpvdz_cart.h5")
    gem = to_AGP(gem)
    return sd, Wavefunction_data(jastrow_data=_jastrow(sd, aos, cpd, dtn=True), geminal_data=gem)


def _fx_agp_ao_sphe_j3_mo():
    sd, aos, mos_up, _md, gem, cpd = _load("H2_ae_ccpvdz_sphe.h5")
    gem_ao = Geminal_data.convert_from_MOs_to_AOs(to_AGP(gem))
    return sd, Wavefunction_data(jastrow_data=_jastrow(sd, aos, cpd, j3_orb=mos_up), geminal_data=gem_ao)


def _fx_sd_open_shell():
    sd, aos, _mu, _md, gem, cpd = _load("N_ae_ccpvdz_cart.h5")
    return sd, Wavefunction_data(jastrow_data=_jastrow(sd, aos, cpd), geminal_data=gem)


def _fx_agpn_open_shell():
    from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd

    sd, aos, _mu, _md, gem, cpd = _load("N_ae_ccpvdz_cart.h5")
    agpn = build_explicit_common_rank_agpn_from_sd(gem)
    return sd, Wavefunction_data(jastrow_data=_jastrow(sd, aos, cpd), geminal_data=agpn)


def _fx_sd_pbc_complex():
    sd, aos, _mu, _md, gem, cpd = _load("H2_pbc_ae_k025_sphe.h5")
    return sd, Wavefunction_data(jastrow_data=_jastrow(sd, aos, cpd), geminal_data=gem)


_FIXTURES = {
    "agp_mo_cart_j123_dtn": (_fx_agp_mo_dtn, {"lambda_matrix", "j3_poly_alpha", "j3_poly_beta", "j3_poly_gamma"}),
    "agp_ao_sphe_j3_on_mos": (_fx_agp_ao_sphe_j3_mo, {"lambda_matrix", "lambda_basis_exp", "j3_matrix"}),
    "sd_open_shell_cart": (_fx_sd_open_shell, {"mo_coefficients_up", "mo_coefficients_dn", "j3_basis_exp"}),
    "agpn_open_shell_cart": (_fx_agpn_open_shell, {"lambda_diag", "lambda_unpaired", "mo_coefficients_dn"}),
    "sd_pbc_complex_sphe": (_fx_sd_pbc_complex, {"mo_coefficients_up", "j3_matrix", "lambda_basis_coeff"}),
}


def _build(name):
    builder, _required = _FIXTURES[name]
    sd, wf = builder()
    gem = wf.geminal_data
    r_up, r_dn = _positions(sd, gem.num_electron_up, gem.num_electron_dn, seed=7)
    return wf, r_up, r_dn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.light
def test_every_wavefunction_pytree_leaf_field_is_reviewed():
    """Every pytree leaf field reachable from Wavefunction_data is in the reviewed table (no stale rows)."""
    check_inventory(Wavefunction_data, _REVIEWED_FIELDS)


@pytest.mark.light
def test_reviewed_variational_fields_are_classified_by_the_mask():
    """Name-masked variational fields carry the "param" classification; positions carry "coord"."""
    check_static_classification(Wavefunction_data, _REVIEWED_FIELDS)


@pytest.mark.light
@pytest.mark.parametrize("fixture_name", sorted(_FIXTURES))
def test_each_flag_off_freezes_exactly_its_variational_blocks(fixture_name):
    """Each optimizer block flows with all flags on and is frozen by (only) its own flag off."""
    wf, r_up, r_dn = _build(fixture_name)
    covered = check_flag_freezes_its_blocks(wf, r_up, r_dn, _REVIEWED_FIELDS)
    missing = _FIXTURES[fixture_name][1] - set(covered)
    assert not missing, f"fixture {fixture_name} no longer exposes the blocks it is meant to cover: {sorted(missing)}"


@struct.dataclass
class _SyntheticBlock:
    jastrow_2b_param: float = struct.field(pytree_node=True, default=1.0)
    new_variational_param: np.ndarray = struct.field(pytree_node=True, default_factory=lambda: np.ones(3))


@struct.dataclass
class _SyntheticRoot:
    block: _SyntheticBlock = struct.field(pytree_node=True, default_factory=_SyntheticBlock)


@pytest.mark.light
def test_inventory_gate_rejects_an_unreviewed_variational_field():
    """Negative control: a new untagged field fails the inventory and, if reviewed as name-masked, the classification."""
    reviewed = {("_SyntheticBlock", "jastrow_2b_param"): ("param-name", "synthetic")}
    check_inventory(_SyntheticRoot, {**reviewed, ("_SyntheticBlock", "new_variational_param"): ("fixed", "control")})
    with pytest.raises(
        AssertionError,
        match=r"unreviewed pytree leaf field\(s\) reachable from _SyntheticRoot: _SyntheticBlock\.new_variational_param",
    ):
        check_inventory(_SyntheticRoot, reviewed)
    with pytest.raises(AssertionError, match=r"stale _REVIEWED_FIELDS entries"):
        check_inventory(
            _SyntheticRoot,
            {
                **reviewed,
                ("_SyntheticBlock", "new_variational_param"): ("fixed", "x"),
                ("_SyntheticBlock", "gone"): ("fixed", "x"),
            },
        )
    as_param = {**reviewed, ("_SyntheticBlock", "new_variational_param"): ("param-name", "control")}
    with pytest.raises(AssertionError, match=r"_SyntheticBlock\.new_variational_param \(role param-name\) is classified None"):
        check_static_classification(_SyntheticRoot, as_param)


@pytest.mark.light
def test_mask_gate_rejects_a_name_list_that_misses_a_variational_field(monkeypatch):
    """Negative control: dropping jastrow_1b_param from _PARAM_FIELD_NAMES fails the static and behavioural checks."""
    wf, r_up, r_dn = _build("agp_mo_cart_j123_dtn")
    monkeypatch.setattr(diff_mask_module, "_PARAM_FIELD_NAMES", set(diff_mask_module._PARAM_FIELD_NAMES) - {"jastrow_1b_param"})
    with pytest.raises(AssertionError, match=r"Jastrow_one_body_data\.jastrow_1b_param \(role param-name\) is classified None"):
        check_static_classification(Wavefunction_data, _REVIEWED_FIELDS)
    with pytest.raises(AssertionError, match=r"block 'j1_param' is not frozen with opt_J1_param=False"):
        check_flag_freezes_its_blocks(wf, r_up, r_dn, _REVIEWED_FIELDS)

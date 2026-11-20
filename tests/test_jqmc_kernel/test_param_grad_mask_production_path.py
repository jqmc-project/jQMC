"""Production-path c-axis parameter-gradient tests.

WHY THIS FILE EXISTS
====================

The production c-axis (variational parameter) gradients -- ``O_k = d ln|Psi| / dc``
and ``de_L / dc`` -- run through the unified ``jqmc.derivatives`` interface with
``analytic=False`` (the autograd twin, numerically the pre-C1 masked autograd)::

    O_k    = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
    dEL_dc = derivatives.compute_grad_e_L (ham, r_up, r_dn, RT, flags, analytic=False)["c"]

``flags`` is the resolved ``opt_flags`` whitelist; the mask / ``stop_gradient``
that freezes the frozen parameter blocks is derived and applied INSIDE
``derivatives`` (the caller never writes ``with_param_grad_mask`` /
``stop_gradient``). The interface returns the RAW ``Wavefunction_data``-shaped
grad for the ``"c"`` axis; the driver collapses it to the flat block-name dict
production stores via :py:func:`jqmc.derivatives.bridge._reduce_c_param_grads`.

This module exercises that production wiring end-to-end:

  * the mask FREEZES every orbital-basis payload leaf in-trace (the RAW grad's
    ``exponents`` / ``coefficients`` / ``spline_coefficients`` / ``phase_is_sin``
    are exactly zero under a geminal-only opt), which blocks an OOM-shaped leak;
  * the requested variational block FLOWS (non-zero finite grad), including the
    ``SD_data`` / ``AGPn_data`` direct-MO ``mo_coefficients_up`` block with the
    closed-shell ``C_dn = conj(C_up)`` tie the reducer reconstructs;
  * the reduced dict carries EXACTLY the active block names.

The matrix runs on every orbital basis the geminal can take:

    * molecular cart GTO  (``H2_ae_ccpvdz_cart.h5`` -> SD_data)
    * molecular sphe GTO  (``H2_ae_ccpvdz_sphe.h5`` -> SD_data)
    * periodic pGTO       (image-expanded one s-shell-per-atom, AGP)
    * periodic PW         (real cos / sin Gamma-point basis, AGP)
    * periodic B-spline   (small 10x10x10 cube, AGP)
    * periodic pGTO + PW  (``AOs_composite_data``, AGP)
    * periodic pGTO + B-spline (``AOs_composite_data`` -- the OOM repro shape)

against BOTH the ``compute_local_energy`` (de_L / dc) path and the
``evaluate_ln_wavefunction`` (dln_Psi / dc) path.

The position-grad (force / stress, r/R axis) leak tests at the bottom exercise
the SEPARATE position-grad path via the PRODUCTION bridge closures
(``bridge.batched_grad_e_L`` / ``batched_grad_ln_Psi`` with ``wrt=("r", "R")``,
``with_geminal_inv=True``) -- the SAME closures ``MCMC.run`` builds through
``_cached_grad_e_L_rR`` / ``_cached_grad_ln_psi_rR``. The bridge returns PURE
position arrays (a drift/force tuple, no wavefunction pytree), so the
"wf-param freeze / no-58GiB-tape" guarantee is STRUCTURAL rather than a
runtime mask check.
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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_MEDIUM, RTOL_BIT_EXACT, AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    AOs_bspline_data,
    AOs_composite_data,
    AOs_plane_wave_data,
    AOs_sphe_gto_data,
    _apply_image_expansion,
)
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
from jqmc.determinant import AGPn_data, Geminal_data, SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_DTN_data,
    Jastrow_data,
    Jastrow_two_body_data,
)
from jqmc import derivatives
from jqmc.derivatives import bridge
from jqmc.derivatives.bridge import _collect_c_param_derivative_sources, _reduce_c_param_grads
from jqmc.jqmc_mcmc import _ANALYTIC_GRAD_ENABLED, _geminal_inv_batched
from jqmc.molecular_orbital import MOs_data
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    evaluate_wavefunction,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


# ---------------------------------------------------------------------------
# PBC dummy Hamiltonian builders.
#
# Adapted from ``tests/test_jqmc_kernel/test_hamiltonian.py`` :py:func:`_build_pbc_dummy_hamiltonian`
# (cubic 4-H cell, 2-up / 2-dn, MO-wrapped AO basis) with DTN added so we have
# a non-orbital "must-flow" sentinel in every gradient check.
# ---------------------------------------------------------------------------

_PBC_A = 6.0
_PBC_BSPLINE_NS = (10, 10, 10)
_PBC_BSPLINE_ORDER = 5


def _pbc_structure(a: float = _PBC_A) -> Structure_data:
    cell = np.eye(3, dtype=np.float64) * a
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5 * a, 0.0, 0.0],
            [0.0, 0.5 * a, 0.0],
            [0.0, 0.0, 0.5 * a],
        ],
        dtype=np.float64,
    )
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("H", "H", "H", "H"),
        atomic_labels=("H1", "H2", "H3", "H4"),
    )


def _pbc_pw_leaf(structure: Structure_data) -> AOs_plane_wave_data:
    recip = structure.recip_cell
    millers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    g_list: list[np.ndarray] = []
    is_sin: list[float] = []
    for m in millers:
        g = m @ recip
        if np.all(m == 0):
            g_list.append(g)
            is_sin.append(0.0)
        else:
            g_list.append(g)
            is_sin.append(0.0)
            g_list.append(g)
            is_sin.append(1.0)
    g_vectors = np.asarray(g_list, dtype=np.float64)
    phase_is_sin = np.asarray(is_sin, dtype=np.float64)
    return AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)


def _pbc_bspline_bands(frac):
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    tp = 2.0 * np.pi
    b0 = np.sin(tp * x) + 0.5 * np.cos(tp * y) + 0.25 * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + 0.3 * np.sin(tp * z)
    return np.stack([b0, b1], axis=0)


def _pbc_bspline_leaf(structure: Structure_data) -> AOs_bspline_data:
    ns = _PBC_BSPLINE_NS
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    frac = np.stack([fx, fy, fz], axis=-1)
    vals = _pbc_bspline_bands(frac)
    coeffs_cube = bspline_prefilter_periodic(vals.astype(np.complex128), _PBC_BSPLINE_ORDER)
    coeffs_flat = coeffs_cube.reshape(coeffs_cube.shape[0], -1)
    return AOs_bspline_data.build(structure, coeffs_flat, ns, _PBC_BSPLINE_ORDER)


def _pbc_pgto_leaf(structure: Structure_data):
    n_atom = len(structure.atomic_numbers)
    base = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=tuple(range(n_atom)),
        num_ao=n_atom,
        num_ao_prim=n_atom,
        orbital_indices=tuple(range(n_atom)),
        exponents=np.full(n_atom, 0.6, dtype=np.float64),
        coefficients=np.ones(n_atom, dtype=np.float64),
        angular_momentums=tuple([0] * n_atom),
        magnetic_quantum_numbers=tuple([0] * n_atom),
    )
    return _apply_image_expansion(base, structure, AO_IMAGE_TRUNCATION_ATOL)


def _pbc_aos(kind: str, structure: Structure_data):
    if kind == "pw":
        return _pbc_pw_leaf(structure)
    if kind == "bspline":
        return _pbc_bspline_leaf(structure)
    if kind == "pgto":
        return _pbc_pgto_leaf(structure)
    if kind == "pgto+pw":
        return _pbc_pgto_leaf(structure) + _pbc_pw_leaf(structure)
    if kind == "pgto+bspline":
        return _pbc_pgto_leaf(structure) + _pbc_bspline_leaf(structure)
    raise ValueError(f"unknown periodic AO kind: {kind}")


def _attach_dtn(structure: Structure_data) -> Jastrow_DTN_data:
    """Build a small DTN block (u + chi + f) so DTN can serve as the must-flow sentinel."""
    return Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=(1.0,),  # all-electron H test structure
        pseudized_per_species=(False,),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        random_init=True,
        seed=2,
    )


def _build_pbc_ham(kind: str, seed: int = 2025, include_dtn: bool = True) -> Hamiltonian_data:
    structure = _pbc_structure()
    aos_data = _pbc_aos(kind, structure)
    aos_data.sanity_check()

    num_mo = 2
    n_up = n_dn = 2
    rng = np.random.default_rng(seed)
    mo_coefficients = rng.standard_normal((num_mo, aos_data.num_ao)).astype(np.float64)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    lambda_paired = rng.uniform(-0.2, 0.2, size=(num_mo, num_mo))
    lambda_paired = 0.5 * (lambda_paired + lambda_paired.T) + np.eye(num_mo)
    lambda_matrix = np.asarray(lambda_paired, dtype=np.float64)

    geminal_data = Geminal_data(
        num_electron_up=n_up,
        num_electron_dn=n_dn,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=lambda_matrix,
    )
    geminal_data.sanity_check()

    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0,
        jastrow_2b_type="pade",
        structure_data=structure,
    )
    if include_dtn:
        jastrow_data = Jastrow_data.build(jastrow_two_body_data=j2, jastrow_dtn_data=_attach_dtn(structure))
    else:
        jastrow_data = Jastrow_data.build(jastrow_two_body_data=j2)
    wf = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)

    coulomb_potential_data = Coulomb_potential_data(structure_data=structure)
    ewald_data = build_ewald_data(structure, coulomb_potential_data, tol=1e-10)
    coulomb_with_ewald = coulomb_potential_data.replace(ewald_data=ewald_data)

    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb_with_ewald,
        wavefunction_data=wf,
    )
    ham.sanity_check()
    return ham


def _build_molecular_ham(trexio_file: str, include_dtn: bool = False) -> Hamiltonian_data:
    sd, _aos, _, _, gem, cpd = read_trexio_file(trexio_file=os.path.join(TREXIO_DIR, trexio_file), store_tuple=True)
    # The closed-shell H2 geminal loads as ``SD_data`` (direct-MO variational
    # block ``mo_coefficients_up``); we KEEP it as SD here (unlike the pre-C1
    # test which promoted to AGP) so the reducer's SD direct-MO + time-reversal
    # tie is exercised on the production path.
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    if include_dtn:
        jastrow_data = Jastrow_data.build(jastrow_two_body_data=j2, jastrow_dtn_data=_attach_dtn(sd))
    else:
        jastrow_data = Jastrow_data.build(jastrow_two_body_data=j2)
    wf = Wavefunction_data(geminal_data=gem, jastrow_data=jastrow_data)
    return Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cpd,
        wavefunction_data=wf,
    )


# ---------------------------------------------------------------------------
# Per-kind fixtures.
# ---------------------------------------------------------------------------

_ORBITAL_KINDS = [
    "mol_gto_cart",
    "mol_gto_sphe",
    "pbc_pgto",
    "pbc_pw",
    "pbc_bspline",
    "pbc_pgto_pw",
    "pbc_pgto_bspline",
]

# Orbital kinds that have GTO ``exponents`` / ``coefficients`` somewhere in the
# tree (used to drive the ``opt_lambda_basis_exp`` / ``coeff`` flow assertions).
_GTO_KINDS = {"mol_gto_cart", "mol_gto_sphe", "pbc_pgto", "pbc_pgto_pw", "pbc_pgto_bspline"}


def _build_ham(kind: str, include_dtn: bool = False) -> Hamiltonian_data:
    if kind == "mol_gto_cart":
        return _build_molecular_ham("H2_ae_ccpvdz_cart.h5", include_dtn=include_dtn)
    if kind == "mol_gto_sphe":
        return _build_molecular_ham("H2_ae_ccpvdz_sphe.h5", include_dtn=include_dtn)
    if kind == "pbc_pgto":
        return _build_pbc_ham("pgto", include_dtn=include_dtn)
    if kind == "pbc_pw":
        return _build_pbc_ham("pw", include_dtn=include_dtn)
    if kind == "pbc_bspline":
        return _build_pbc_ham("bspline", include_dtn=include_dtn)
    if kind == "pbc_pgto_pw":
        return _build_pbc_ham("pgto+pw", include_dtn=include_dtn)
    if kind == "pbc_pgto_bspline":
        return _build_pbc_ham("pgto+bspline", include_dtn=include_dtn)
    raise ValueError(f"unknown orbital kind: {kind}")


def _pick_electron_positions(ham: Hamiltonian_data, kind: str, seed: int = 11, *, return_attempts: bool = False):
    """Sample positions where ``|Psi|`` is non-zero AND electrons are close to
    nuclei (within ~1 Bohr) so the DTN ``chi``/``f`` channels actually carry a
    gradient (the must-flow sentinel)."""
    rng = np.random.default_rng(seed)
    geminal = ham.wavefunction_data.geminal_data
    n_up = geminal.num_electron_up
    n_dn = geminal.num_electron_dn
    centers = np.asarray(ham.structure_data.positions)
    for attempt in range(100):
        r_up = centers[rng.integers(0, centers.shape[0], size=n_up)] + 0.6 * rng.standard_normal((n_up, 3))
        r_dn = centers[rng.integers(0, centers.shape[0], size=n_dn)] + 0.6 * rng.standard_normal((n_dn, 3))
        r_up = r_up.astype(np.float64)
        r_dn = r_dn.astype(np.float64)
        psi = float(evaluate_wavefunction(ham.wavefunction_data, jnp.asarray(r_up), jnp.asarray(r_dn)))
        if abs(psi) > 1e-8 and np.isfinite(psi):
            positions = (jnp.asarray(r_up), jnp.asarray(r_dn))
            return (*positions, attempt + 1) if return_attempts else positions
    pytest.fail(
        f"no node-far configuration found for {kind} with the fixed seed; this precondition"
        " search is deterministic and a failure indicates evaluate_wavefunction regressed"
        " (e.g. returning 0/NaN), not an environment fluke"
    )


@pytest.fixture(scope="module", params=_ORBITAL_KINDS)
def fixture(request):
    """Parametrise across all geminal-orbital backings (module-scoped)."""
    kind = request.param
    ham = _build_ham(kind)
    r_up, r_dn = _pick_electron_positions(ham, kind)
    RT = jnp.eye(3, dtype=jnp.float64)
    return kind, ham, r_up, r_dn, RT


@pytest.fixture(autouse=True)
def _clear_jit_cache_between_tests():
    """Clear the JAX cache between tests (see pre-C1 note: cross-fixture
    ``pytree_node=False`` numpy-array metadata collides in the jit cache key)."""
    yield
    jax.clear_caches()


# ---------------------------------------------------------------------------
# Flag vocabulary + C1 production-wiring helpers.
#
# These call the EXACT interface + reducer ``MCMC.run()`` uses on the C1 c-axis
# path: ``derivatives.compute_grad_{ln_Psi,e_L}(..., flags, analytic=False)["c"]``
# vmapped over a one-walker batch, then ``_reduce_c_param_grads``. The result
# reads as (raw single-walker grad pytree, flat block-name dict).
# ---------------------------------------------------------------------------

_BASE_FLAGS = dict(
    opt_J1_param=False,
    opt_J2_param=False,
    opt_J3_param=False,
    opt_geminal=False,
    opt_J3_basis_exp=False,
    opt_J3_basis_coeff=False,
    opt_lambda_basis_exp=False,
    opt_lambda_basis_coeff=False,
    opt_JDTN_u_param=False,
    opt_JDTN_chi_param=False,
    opt_JDTN_f_param=False,
)


@pytest.mark.light
@pytest.mark.parametrize("geminal", ["SD", "AGPn", "AGP_MO"])
@pytest.mark.parametrize(
    "opt_geminal,opt_exp,opt_coeff",
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Linear probe of raw orbital leaves; the verdict evaluates no electron coordinates."
)
def test_mo_and_radial_mask_flags_are_independent(geminal, opt_geminal, opt_exp, opt_coeff):
    """Every MO/exponent/contraction flag combination gives exact zero/one leaf sensitivities.

    A linear probe of each raw orbital leaf isolates masking from the scientific
    forward model. SD/AGPn expose MO parameters; dense AGP over fixed MOs does not.
    Both spin leaves and every radial element are checked, including all-disabled.
    """
    from tests.test_jqmc_kernel._generated_matrix_base import _apply_gem

    ham = _apply_gem(_build_ham("mol_gto_cart"), geminal)
    flags = _resolve_flags(opt_geminal=opt_geminal, opt_lambda_basis_exp=opt_exp, opt_lambda_basis_coeff=opt_coeff)

    def probe(wf):
        masked = wf.with_param_grad_mask(**flags)
        total = 0.0
        for orb in (masked.geminal_data.orb_data_up_spin, masked.geminal_data.orb_data_dn_spin):
            total = (
                total
                + jnp.sum(orb.mo_coefficients)
                + jnp.sum(orb.aos_data.radial.exponents)
                + jnp.sum(orb.aos_data.radial.coefficients)
            )
        return total

    grad = jax.jit(jax.grad(probe))(ham.wavefunction_data)
    for orb in (grad.geminal_data.orb_data_up_spin, grad.geminal_data.orb_data_dn_spin):
        for value, enabled in (
            (orb.mo_coefficients, opt_geminal and geminal in ("SD", "AGPn")),
            (orb.aos_data.radial.exponents, opt_exp),
            (orb.aos_data.radial.coefficients, opt_coeff),
        ):
            np.testing.assert_array_equal(np.asarray(value), np.full_like(value, int(enabled)))


def _resolve_flags(**overrides) -> dict:
    flags = dict(_BASE_FLAGS)
    flags.update(overrides)
    return flags


# opt_* flag -> production ``param_grad_flags`` block name (used to mirror the
# run()-side ``if not param_grad_flags.get(name, True): continue`` filter, which
# drops the zero-valued frozen blocks the reducer emits via ``collect_param_grads``).
_OPT_TO_BLOCK = {
    "opt_J1_param": "j1_param",
    "opt_J2_param": "j2_param",
    "opt_J3_param": "j3_matrix",
    "opt_geminal": "lambda_matrix",
    "opt_J3_basis_exp": "j3_basis_exp",
    "opt_J3_basis_coeff": "j3_basis_coeff",
    "opt_lambda_basis_exp": "lambda_basis_exp",
    "opt_lambda_basis_coeff": "lambda_basis_coeff",
    "opt_JDTN_u_param": "j3_poly_alpha",
    "opt_JDTN_chi_param": "j3_poly_beta",
    "opt_JDTN_f_param": "j3_poly_gamma",
}


def _stored_filter(flat: dict, flags: dict) -> dict:
    """Mirror MCMC.run()'s downstream filter: keep block ``name`` iff
    ``param_grad_flags.get(name, True)`` -- blocks with a mapped opt flag stay
    only when that flag is ``True``; the direct-MO geminal blocks
    (``mo_coefficients_up`` etc., not in the flag map) are always kept."""
    block_active = {blk: bool(flags.get(opt, False)) for opt, blk in _OPT_TO_BLOCK.items()}
    return {k: v for k, v in flat.items() if block_active.get(k, True)}


def _c_grad_ln_psi(ham, r_up, r_dn, **flags_overrides):
    flags = _resolve_flags(**flags_overrides)
    graw_batched = jax.jit(
        jax.vmap(
            lambda ru, rd: derivatives.compute_grad_ln_Psi(ham, ru, rd, flags, analytic=False)["c"],
            in_axes=(0, 0),
        )
    )(jnp.expand_dims(r_up, 0), jnp.expand_dims(r_dn, 0))
    flat = _stored_filter(_reduce_c_param_grads(ham.wavefunction_data, graw_batched, 1, flags), flags)
    graw = jax.tree_util.tree_map(lambda x: x[0] if x is not None else None, graw_batched)
    return graw, {k: np.asarray(v)[0] for k, v in flat.items()}


def _c_grad_e_L(ham, r_up, r_dn, RT, **flags_overrides):
    flags = _resolve_flags(**flags_overrides)
    graw_batched = jax.jit(
        jax.vmap(
            lambda ru, rd, rt: derivatives.compute_grad_e_L(ham, ru, rd, rt, flags, analytic=False)["c"],
            in_axes=(0, 0, 0),
        )
    )(jnp.expand_dims(r_up, 0), jnp.expand_dims(r_dn, 0), jnp.expand_dims(RT, 0))
    flat = _stored_filter(_reduce_c_param_grads(ham.wavefunction_data, graw_batched, 1, flags), flags)
    graw = jax.tree_util.tree_map(lambda x: x[0] if x is not None else None, graw_batched)
    return graw, {k: np.asarray(v)[0] for k, v in flat.items()}


def _sources_from_single_raw(ham, graw, **flags_overrides):
    """Build the exact C6 payload from an unbatched test-side raw gradient."""
    flags = _resolve_flags(**flags_overrides)
    blocks = ham.wavefunction_data.get_variational_blocks(
        state_token=("c6-production-path", 0),
        **flags,
    )
    graw_batched = jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0) if value is not None else None, graw)
    return _collect_c_param_derivative_sources(ham.wavefunction_data, graw_batched, blocks, 1, flags)


# ---------------------------------------------------------------------------
# Leaf-walking helpers.
# ---------------------------------------------------------------------------


def _mx(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        return float("nan")
    return float(np.max(np.abs(arr)))


def _basis_payload_maxes(orb) -> dict[str, float]:
    """Recurse into a grad-pytree orbital subtree and return |grad|_inf for
    every NON-MO basis payload leaf (``exponents`` / ``coefficients`` /
    ``spline_coefficients`` / ``phase_is_sin``). ``mo_coefficients`` is excluded
    -- it is the SD/AGPn variational block, so it may legitimately flow."""
    out: dict[str, float] = {}

    def _recurse(node, path):
        if node is None:
            return
        for name in ("exponents", "coefficients", "spline_coefficients", "phase_is_sin"):
            if hasattr(node, name) and getattr(node, name) is not None:
                out[f"{path}{name}"] = _mx(getattr(node, name))
        if hasattr(node, "aos_data"):
            _recurse(node.aos_data, path + "aos_data.")
        if isinstance(node, AOs_composite_data):
            for i, child in enumerate(node.children):
                _recurse(child, path + f"children[{i}].")

    _recurse(orb, "")
    return out


def _geminal_basis_payload_maxes(grad_pytree_wf) -> dict[str, float]:
    """Return ``|grad|_inf`` for every non-MO basis-payload leaf in the geminal orbitals."""
    geminal = grad_pytree_wf.geminal_data
    out: dict[str, float] = {}
    for spin, orb in (("up", geminal.orb_data_up_spin), ("dn", geminal.orb_data_dn_spin)):
        for k, v in _basis_payload_maxes(orb).items():
            out[f"{spin}.{k}"] = v
    return out


def _geminal_block_name(ham: Hamiltonian_data) -> str:
    """The reduced-dict block name a geminal-only opt (``opt_geminal=True``)
    produces: SD/AGPn expose ``mo_coefficients_up``; legacy AGP the dense
    ``lambda_matrix``."""
    gem = ham.wavefunction_data.geminal_data
    if isinstance(gem, (SD_data, AGPn_data)):
        return "mo_coefficients_up"
    return "lambda_matrix"


# ===========================================================================
# C1 c-axis (variational parameter) gradient tests.
# ===========================================================================


class TestCAxisLight:
    """Light (per-commit) smoke of the C1 c-axis path on a representative
    molecular SD_data fixture + a periodic AGP fixture -- the heavy class below
    sweeps the full 7-backing matrix."""

    @pytest.mark.light
    def test_mol_sd_ln_psi_and_e_L_geminal_only(self):
        """Mol sd ln psi and e L geminal only."""
        ham = _build_ham("mol_gto_cart")
        r_up, r_dn = _pick_electron_positions(ham, "mol_gto_cart")
        RT = jnp.eye(3, dtype=jnp.float64)

        for tag, (graw, flat) in (
            ("ln", _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=True)),
            ("eL", _c_grad_e_L(ham, r_up, r_dn, RT, opt_geminal=True)),
        ):
            assert set(flat.keys()) == {"mo_coefficients_up"}, f"[{tag}] keys {set(flat.keys())}"
            assert _mx(flat["mo_coefficients_up"]) > 0.0 and np.isfinite(_mx(flat["mo_coefficients_up"]))
            leaks = {k: v for k, v in _geminal_basis_payload_maxes(graw).items() if v != 0.0}
            assert not leaks, f"[{tag}] basis payload leaked: {leaks}"

    @pytest.mark.light
    def test_mol_sd_jastrow_only_freezes_geminal(self):
        """Mol sd jastrow only freezes geminal."""
        ham = _build_ham("mol_gto_cart")
        r_up, r_dn = _pick_electron_positions(ham, "mol_gto_cart")
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_J2_param=True)
        assert set(flat.keys()) == {"j2_param"}
        assert _mx(graw.geminal_data.orb_data_up_spin.mo_coefficients) == 0.0

    @pytest.mark.light
    def test_mol_sd_closed_shell_real_tie_publishes_one_full_direct_column(self):
        """Mol sd closed shell real tie publishes one full direct column."""
        ham = _build_ham("mol_gto_cart")
        r_up, r_dn = _pick_electron_positions(ham, "mol_gto_cart")
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=True)
        gu = np.asarray(graw.geminal_data.orb_data_up_spin.mo_coefficients)
        gd = np.asarray(graw.geminal_data.orb_data_dn_spin.mo_coefficients)
        sources = _sources_from_single_raw(ham, graw, opt_geminal=True)
        assert len(sources) == 1
        source = sources[0]
        assert source.name == "mo_coefficients_up"
        assert source.d_p is None and source.d_pbar is None
        np.testing.assert_allclose(source.direct[0], gu + gd, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_MEDIUM)
        np.testing.assert_allclose(
            np.real(source.direct[0]),
            np.asarray(flat["mo_coefficients_up"]),
            rtol=RTOL_BIT_EXACT,
            atol=ATOL_PYTHON_FP64_MEDIUM,
        )

    @pytest.mark.light
    def test_pbc_agp_geminal_only(self):
        """PBC agp geminal only."""
        ham = _build_pbc_ham("pgto", include_dtn=False)
        r_up, r_dn = _pick_electron_positions(ham, "pbc_pgto")
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=True)
        assert set(flat.keys()) == {"lambda_matrix"}
        assert _mx(flat["lambda_matrix"]) > 0.0 and np.isfinite(_mx(flat["lambda_matrix"]))
        leaks = {k: v for k, v in _geminal_basis_payload_maxes(graw).items() if v != 0.0}
        assert not leaks, f"basis payload leaked: {leaks}"


class TestCAxisProductionPath:
    @pytest.mark.heavy
    def test_geminal_only_freezes_orbital_basis_ln_psi(self, fixture):
        """``opt_geminal=True`` only, via ``compute_grad_ln_Psi(...)["c"]``.

        The requested geminal block flows; every orbital BASIS payload leaf
        (exponents / coefficients / spline_coefficients / phase_is_sin) in the
        RAW grad is frozen to exactly zero -- the mask (applied in-trace inside
        ``derivatives``) blocks the OOM-shaped basis tape.
        """
        kind, ham, r_up, r_dn, _RT = fixture
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=True)

        block = _geminal_block_name(ham)
        assert set(flat.keys()) == {block}, f"[{kind}] reduced keys {set(flat.keys())} != {{{block!r}}}"
        assert _mx(flat[block]) > 0.0 and np.isfinite(_mx(flat[block])), (
            f"[{kind}] geminal block {block} must carry non-zero finite O_k; |grad|={_mx(flat[block])}"
        )

        leaks = {k: v for k, v in _geminal_basis_payload_maxes(graw).items() if v != 0.0}
        assert not leaks, (
            f"[{kind}] orbital basis payload leaked under geminal-only opt (mask failed to freeze in-trace): {leaks}"
        )

    @pytest.mark.heavy
    def test_geminal_only_freezes_orbital_basis_e_L(self, fixture):
        """Same as above but through ``compute_grad_e_L(...)["c"]`` (de_L/dc).

        This is the user's silicon pGTO + B-spline OOM shape on the de_L/dc
        path (kinetic-energy Laplacian tape over the orbital basis).
        """
        kind, ham, r_up, r_dn, RT = fixture
        graw, flat = _c_grad_e_L(ham, r_up, r_dn, RT, opt_geminal=True)

        block = _geminal_block_name(ham)
        assert set(flat.keys()) == {block}, f"[{kind}] reduced keys {set(flat.keys())} != {{{block!r}}}"
        assert _mx(flat[block]) > 0.0 and np.isfinite(_mx(flat[block])), (
            f"[{kind}] geminal block {block} must carry non-zero finite de_L/dc; |grad|={_mx(flat[block])}"
        )

        leaks = {k: v for k, v in _geminal_basis_payload_maxes(graw).items() if v != 0.0}
        assert not leaks, f"[{kind}] orbital basis payload leaked under geminal-only opt on de_L/dc path: {leaks}"

    @pytest.mark.heavy
    def test_jastrow_only_freezes_geminal_and_basis(self, fixture):
        """``opt_J2_param=True`` only: J2 flows; geminal MO + orbital basis frozen.

        The J2 Jastrow parameter is present in every fixture. Under J2-only opt
        the reduced dict is exactly ``{"j2_param"}`` and neither the geminal
        MO block nor any basis payload leaf appears / flows.
        """
        kind, ham, r_up, r_dn, _RT = fixture
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_J2_param=True)

        assert set(flat.keys()) == {"j2_param"}, f"[{kind}] reduced keys {set(flat.keys())} != {{'j2_param'}}"
        assert np.isfinite(_mx(flat["j2_param"])), f"[{kind}] j2_param grad not finite"

        gem = graw.geminal_data
        mo_up = getattr(gem.orb_data_up_spin, "mo_coefficients", None)
        if mo_up is not None:
            assert _mx(mo_up) == 0.0, f"[{kind}] mo_coefficients leaked under J2-only opt: {_mx(mo_up)}"
        leaks = {k: v for k, v in _geminal_basis_payload_maxes(graw).items() if v != 0.0}
        assert not leaks, f"[{kind}] orbital basis payload leaked under J2-only opt: {leaks}"

    @pytest.mark.heavy
    @pytest.mark.parametrize("kind", ["mol_gto_cart", "mol_gto_sphe", "pbc_pgto"])
    def test_lambda_basis_flows_gto_exp_coeff_keeps_mo_frozen(self, kind):
        """Basis-only optimization differentiates GTO exponents/contractions, not MO coefficients.

        With opt_geminal disabled, both direct-MO and dense-lambda parameters
        remain frozen while the independently requested radial blocks flow.

        Parametrized over the PURE-GTO kinds ONLY, by construction rather than
        by skip: PW / B-spline leaves carry no exp/coeff to optimise, and the
        composite kinds' partial basis-opt is not implemented in
        ``with_param_grad_mask`` (tracked separately) -- so four of the shared
        fixture's seven kinds could never run this test in any session, and the
        module receipt closure refuses to close such permanently-skipped pins.
        This test therefore builds its own ham per runnable kind instead of
        consuming the seven-kind shared fixture.
        """
        ham = _build_ham(kind)
        r_up, r_dn = _pick_electron_positions(ham, kind)
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=False, opt_lambda_basis_exp=True, opt_lambda_basis_coeff=True)

        assert not any(k == "mo_coefficients" for k in flat), f"[{kind}] bare mo_coefficients leaked: {list(flat)}"
        basis_blocks = [k for k in flat if "basis" in k]
        assert basis_blocks and any(_mx(flat[k]) > 0.0 for k in basis_blocks), (
            f"[{kind}] expected GTO basis (exp/coeff) grads to flow; got keys {list(flat)}"
        )
        assert set(flat) == {"lambda_basis_exp", "lambda_basis_coeff"}
        for orb in (graw.geminal_data.orb_data_up_spin, graw.geminal_data.orb_data_dn_spin):
            np.testing.assert_array_equal(np.asarray(orb.mo_coefficients), np.zeros_like(orb.mo_coefficients))

    @pytest.mark.heavy
    @pytest.mark.parametrize("kind", ["mol_gto_cart", "mol_gto_sphe", "pbc_pgto"])
    def test_joint_geminal_basis_flows_all_requested_blocks(self, kind):
        """Combined geminal/radial flags preserve nonzero determinant and both basis gradients."""
        ham = _build_ham(kind)
        r_up, r_dn = _pick_electron_positions(ham, kind)
        _graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=True, opt_lambda_basis_exp=True, opt_lambda_basis_coeff=True)
        geminal_block = (
            "mo_coefficients_up" if isinstance(ham.wavefunction_data.geminal_data, (SD_data, AGPn_data)) else "lambda_matrix"
        )
        assert set(flat) == {geminal_block, "lambda_basis_exp", "lambda_basis_coeff"}
        for block in flat:
            assert np.all(np.isfinite(flat[block])) and _mx(flat[block]) > 0.0, block

    @pytest.mark.heavy
    def test_dtn_only_freezes_orbital_basis_e_L_single_fixture(self):
        """DTN-only opt + ``compute_grad_e_L(...)["c"]``: direct cover of the
        silicon pGTO + B-spline OOM repro shape on the de_L/dc path.

        Non-parametric because ``Jastrow_DTN_data`` carries ``pytree_node=False``
        ``np.ndarray`` static fields that crash the jit cache-key equality check
        when two DTN instances share a slot across fixtures.
        """
        ham = _build_ham("mol_gto_cart", include_dtn=True)
        r_up, r_dn = _pick_electron_positions(ham, "mol_gto_cart")
        RT = jnp.eye(3, dtype=jnp.float64)
        graw, flat = _c_grad_e_L(ham, r_up, r_dn, RT, opt_JDTN_u_param=True, opt_JDTN_chi_param=True, opt_JDTN_f_param=True)

        expected = {"j3_poly_alpha", "j3_poly_beta", "j3_poly_gamma"}
        assert set(flat.keys()) == expected, f"reduced keys {set(flat.keys())} != {expected} for DTN-only opt"
        assert any(_mx(v) > 0.0 and np.isfinite(_mx(v)) for v in flat.values()), (
            f"expected non-zero finite DTN gradient; maxes={ {k: _mx(v) for k, v in flat.items()} }"
        )
        leaks = {k: v for k, v in _geminal_basis_payload_maxes(graw).items() if v != 0.0}
        assert not leaks, f"orbital basis payload leaked under DTN-only opt on de_L/dc path: {leaks}"

    @pytest.mark.heavy
    def test_closed_shell_mo_tie_publishes_exact_direct_source(self):
        """Gamma-point closed-shell SD publishes the complete real-direction
        derivative without a storage-space optimizer adapter.
        """
        ham = _build_ham("mol_gto_cart")
        assert isinstance(ham.wavefunction_data.geminal_data, SD_data)
        assert bool(ham.wavefunction_data.geminal_data.time_reversal)
        r_up, r_dn = _pick_electron_positions(ham, "mol_gto_cart")
        graw, flat = _c_grad_ln_psi(ham, r_up, r_dn, opt_geminal=True)

        gu = np.asarray(graw.geminal_data.orb_data_up_spin.mo_coefficients)
        gd = np.asarray(graw.geminal_data.orb_data_dn_spin.mo_coefficients)
        sources = _sources_from_single_raw(ham, graw, opt_geminal=True)
        source = sources[0]
        assert source.direct is not None and source.d_p is None and source.d_pbar is None
        np.testing.assert_allclose(source.direct[0], gu + gd, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_MEDIUM)
        np.testing.assert_allclose(
            np.real(source.direct[0]),
            np.asarray(flat["mo_coefficients_up"]),
            rtol=RTOL_BIT_EXACT,
            atol=ATOL_PYTHON_FP64_MEDIUM,
        )


# ===========================================================================
# Position-grad (force / stress) leak tests -- PRODUCTION BRIDGE ROUTE.
#
# These exercise the r/R position-grad closures MCMC.run() actually drives
# (``bridge.batched_grad_e_L`` / ``batched_grad_ln_Psi`` with ``wrt=("r", "R")``,
# ``with_geminal_inv=True``, ``analytic=_ANALYTIC_GRAD_ENABLED``). The bridge
# closure returns a tuple of PURE position arrays -- there is no wavefunction
# pytree to leak wf-param leaves into, so the old "no-58GiB-tape" mask guarantee
# is now STRUCTURAL (asserted below: every returned leaf is a plain jax.Array of
# a position axis, never a Wavefunction_data node).
# ===========================================================================


def _batched_ginv(ham, r_up_b, r_dn_b):
    """Batched running-SM geminal inverse for the ``with_geminal_inv=True`` bridge
    closures (production shape -- the driver reuses the running SM inverse)."""
    return _geminal_inv_batched(ham.wavefunction_data.geminal_data, r_up_b, r_dn_b)[1]


def _production_position_grad_e_L(ham, r_up, r_dn, RT):
    """Production bridge HF-force closure for a one-walker batch. Returns the
    ``(de_L/dr_up, de_L/dr_dn, de_L/dR)`` tuple of pure position arrays -- the
    fused drift + Hellmann-Feynman/Pulay force emitted in ONE trace. ``de_L/dR``
    has shape ``(n_atom, 3)``."""
    r_up_b, r_dn_b, RT_b = jnp.expand_dims(r_up, 0), jnp.expand_dims(r_dn, 0), jnp.expand_dims(RT, 0)
    ginv = _batched_ginv(ham, r_up_b, r_dn_b)
    closure = bridge.batched_grad_e_L(wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
    dr_up_b, dr_dn_b, dR_b = closure(ham, r_up_b, r_dn_b, RT_b, ginv)
    return dr_up_b[0], dr_dn_b[0], dR_b[0]


def _production_position_grad_ln_psi(ham, r_up, r_dn):
    """Production bridge Pulay ln|Psi| closure for a one-walker batch. Returns the
    ``(dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn)`` tuple of pure position
    arrays (Pulay force first). ``dln|Psi|/dR`` has shape ``(n_atom, 3)``."""
    r_up_b, r_dn_b = jnp.expand_dims(r_up, 0), jnp.expand_dims(r_dn, 0)
    ginv = _batched_ginv(ham, r_up_b, r_dn_b)
    closure = bridge.batched_grad_ln_Psi(wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
    dR_b, dr_up_b, dr_dn_b = closure(ham, r_up_b, r_dn_b, ginv)
    return dR_b[0], dr_up_b[0], dr_dn_b[0]


class TestPositionGradProductionPath:
    @pytest.mark.heavy
    def test_position_grad_mask_freezes_wf_params_e_L(self, fixture):
        """Bridge de_L r/R closure: pure position-array tuple, no wf-leaf output.

        Structural no-wf-leak (i): every returned leaf is a plain ``jax.Array`` of
        a position axis -- the bridge never emits a wavefunction pytree, so no
        wf-param leaf can leak. (i) the ``de_L/dR`` force is finite + NON-ZERO with
        shape ``(n_atom, 3)`` (Hellmann-Feynman + Pulay); (ii) at least one
        electron-position (de_L/dr_up, de_L/dr_dn) grad is non-zero.
        """
        kind, ham, r_up, r_dn, RT = fixture
        dr_up, dr_dn, dR = _production_position_grad_e_L(ham, r_up, r_dn, RT)

        # (iii) STRUCTURAL no-wf-leak: pure position arrays, never a wf pytree.
        for name, arr in (("de_L/dr_up", dr_up), ("de_L/dr_dn", dr_dn), ("de_L/dR", dR)):
            assert isinstance(arr, jax.Array), f"[{kind}] {name} is not a plain position array: {type(arr).__name__}"

        n_atom = ham.structure_data.positions.shape[0]
        assert dR.shape == (n_atom, 3), f"[{kind}] expected de_L/dR shape ({n_atom}, 3), got {dR.shape}"
        dR_max = _mx(dR)
        assert dR_max > 0.0 and np.isfinite(dR_max), (
            f"[{kind}] expected total dE_L/dR to be non-zero (Hellmann-Feynman + Pulay), got max |grad|_inf = {dR_max}"
        )
        assert _mx(dr_up) > 0.0 or _mx(dr_dn) > 0.0, (
            f"[{kind}] expected at least one of (de_L/dr_up, de_L/dr_dn) drift to flow; "
            f"got max|dr_up|={_mx(dr_up)} max|dr_dn|={_mx(dr_dn)}"
        )

    @pytest.mark.heavy
    def test_position_grad_mask_freezes_wf_params_ln_psi(self, fixture):
        """Bridge ln|Psi| r/R closure: pure position-array tuple, no wf-leaf output.

        (iii) STRUCTURAL no-wf-leak: every returned leaf is a plain ``jax.Array``
        of a position axis. The Pulay ``dln|Psi|/dR`` has shape ``(n_atom, 3)`` and
        is finite -- but (unlike the de_L/dR force, which carries the always-non-zero
        Hellmann-Feynman term) it is legitimately EXACTLY ZERO for non-atom-centered
        orbital bases (PW / B-spline: moving a nucleus does not move the orbitals),
        so only finiteness is asserted here (matching the original intent). (ii) at
        least one electron-position (r_up, r_dn) grad flows.
        """
        kind, ham, r_up, r_dn, _RT = fixture
        dR, dr_up, dr_dn = _production_position_grad_ln_psi(ham, r_up, r_dn)

        # (iii) STRUCTURAL no-wf-leak: pure position arrays, never a wf pytree.
        for name, arr in (("dln_Psi/dR", dR), ("dln_Psi/dr_up", dr_up), ("dln_Psi/dr_dn", dr_dn)):
            assert isinstance(arr, jax.Array), f"[{kind}] {name} is not a plain position array: {type(arr).__name__}"

        n_atom = ham.structure_data.positions.shape[0]
        assert dR.shape == (n_atom, 3), f"[{kind}] expected dln_Psi/dR shape ({n_atom}, 3), got {dR.shape}"
        assert np.isfinite(_mx(dR)), f"[{kind}] Pulay dln|Psi|/dR must be finite, got max |grad|_inf = {_mx(dR)}"
        assert _mx(dr_up) > 0.0 or _mx(dr_dn) > 0.0, (
            f"[{kind}] expected at least one of (r_up, r_dn) grad to flow; "
            f"got max|grad_r_up|={_mx(dr_up)} max|grad_r_dn|={_mx(dr_dn)}"
        )

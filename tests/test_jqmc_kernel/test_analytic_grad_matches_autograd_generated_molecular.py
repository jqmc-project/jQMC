"""Generated route-fidelity matrix -- MOLECULAR (``bc == mol``) split.

The machine-generated covering-array matrix (see ``_generated_matrix_base``) partitioned by
boundary condition for CI parallelism: this file hosts the LIGHT molecular cells (cart/sphe GTO
SD / AGPn / dense-AO-AGP / MO-AGP). It also hosts the light determinism sanity test and the
co-located analytic-vs-oracle tests (all molecular).

The periodic (``bc == pbc_gamma``) and twist (``bc == twist``) heavy tails live in the sibling
``_periodic`` / ``_twist`` split files; the UNION of the three is byte-identical to the pre-split
monolith.
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
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc import derivatives
from jqmc.determinant import AGP_data, Geminal_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel._generated_matrix_base import (
    _AXES,
    _AGP_MO_AO_FAMILIES,
    _GEMS,
    _QUANTITIES,
    _ao_families,
    _build_ham_for_cell,
    _cell_id,
    _clear_bridge_caches,  # noqa: F401  (autouse fixture, re-exported for collection)
    _generate_matrix,
    _make_j3,
    _resolve,
    _split_cell,
    make_inventory_report,
    run_matrix_cell,
)

# MOLECULAR subset: cell[3] is ALWAYS the bc field (legacy 4-tuple or 5-tuple), so filter on it.
_MATRIX = [(c, h) for c, h in _generate_matrix() if c[3] == "mol"]
assert _MATRIX, "molecular (bc == mol) subset of the generated matrix must be non-empty"

_CELL_PARAMS = [
    pytest.param(cell, marks=(pytest.mark.heavy if heavy else pytest.mark.light), id=_cell_id(cell)) for cell, heavy in _MATRIX
]

_matrix_report = make_inventory_report(_MATRIX, "molecular")  # session-scoped autouse inventory


@pytest.mark.parametrize("axis", _AXES)
@pytest.mark.parametrize("quantity", _QUANTITIES)
@pytest.mark.parametrize("cell", _CELL_PARAMS)
def test_matrix_cell(cell, quantity, axis):
    receipt = run_matrix_cell(cell, quantity, axis)
    assert receipt.status == "passed"


@pytest.mark.light
def test_matrix_generation_deterministic():
    """Two consecutive generations produce the byte-identical cell list (determinism proof)."""
    a, b = _generate_matrix(), _generate_matrix()
    assert a == b
    ha = hashlib.sha256(repr([tuple(c) for c, _ in a]).encode()).hexdigest()
    hb = hashlib.sha256(repr([tuple(c) for c, _ in b]).encode()).hexdigest()
    assert ha == hb, "cell-list generation is non-deterministic"
    print(f"\n[matrix] deterministic cell list: {len(a)} cells, sha256={ha}")


@pytest.mark.light
def test_mo_agp_fixture_axis_is_explicit_and_buildable():
    """The AGP/MO production representation is a real matrix axis, not a label alias."""
    assert _GEMS == ("SD", "AGPn", "AGP", "AGP_MO")
    mo_agp_cells = [cell for cell, _heavy in _generate_matrix() if _split_cell(cell)[4] == "AGP_MO"]
    # AGP_MO is admitted on the atom-centred Gaussian families only: on the QE-projection
    # leaves the whitened lambda is invisible to e_L, so those cells certify nothing about
    # the very parameter they exist for (gdd.engine.derivative_matrix._agp_mo_constructible).
    # Equality, not containment -- narrowing the admitted set further must fail here.
    assert {cell[0] for cell in mo_agp_cells} == set(_AGP_MO_AO_FAMILIES)
    assert set(_AGP_MO_AO_FAMILIES) < set(_ao_families()), "the admitted set must stay a strict subset"
    assert all(_resolve(cell) is not None for cell in mo_agp_cells)

    representative = next(cell for cell in mo_agp_cells if cell[0] == "cart_gto" and cell[3] == "mol")
    stem, with_jastrow, spec, gem_token = _resolve(representative)
    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem_token)
    geminal = ham.wavefunction_data.geminal_data
    assert isinstance(geminal, AGP_data)
    assert isinstance(geminal.orb_data_up_spin, MOs_data)
    assert isinstance(geminal.orb_data_dn_spin, MOs_data)
    assert geminal.orb_data_up_spin.num_mo >= 2
    paired = np.asarray(geminal.lambda_matrix)[:, : geminal.orb_data_dn_spin.num_mo]
    assert np.any(np.abs(paired - np.diag(np.diag(paired))) > 0.0)
    assert geminal.lambda_matrix.shape[0] == geminal.orb_data_up_spin.num_mo
    assert geminal.lambda_matrix.shape[1] == geminal.orb_data_dn_spin.num_mo + (
        geminal.num_electron_up - geminal.num_electron_dn
    )


# --------------------------------------------------------------------------- #
# Co-located analytic-vs-oracle tests.
# These cover two combinations the GENERATED matrix does NOT reach:
#   * an MO-backed three-body Jastrow (identity ``MOs_data`` leaf) on a MOLECULAR GTO
#     leaf + plain SD geminal, on the R / eps (Pulay force / stress) + r (drift) axes.
#     The matrix only builds an MO-wrapped J3 on the EXOTIC composite / plane-wave /
#     B-spline leaves (``_J3_KIND``), never on a molecular GTO leaf, so the geminal-
#     independent MO->bare-AO J3 fold (``assemble._j3_mo_to_bare_ao``) is exercised here
#     directly against its ``_auto`` twin on the molecular repro.
#   * MPC (a ctx-driven correction threaded via ``mpc_ctx``, explicitly OUT OF SCOPE for
#     the matrix -- see the base module docstring): the dense AO-AGP c-axis with MPC on ln_Psi
#     and e_L, plus the rank-robust density-matrix cotangent anchored vs central FD at the
#     exact AGP==SD degeneracy (an FD arbitration the matrix never performs).
# The coverage is the real behavioral run (analytic path + ``_auto`` / FD compare), not a
# registry ``is_analytic`` pre-check.
# --------------------------------------------------------------------------- #
_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


def _draw(gd, seed: int = 0):
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(nu, 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(nd, 3))),
    )


def _build_mo_j3_ham() -> Hamiltonian_data:
    """A plain MO/SD geminal (molecular H2 cart) + an ACTIVE three-body Jastrow whose orbital
    leaf is an identity ``MOs_data`` -- the MO-backed J3 repro on a MOLECULAR GTO leaf that the
    generated matrix never builds (it MO-wraps only the exotic composite/PW/bspline leaves).
    """
    (structure_data, aos_data, _mu, _md, geminal_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, "H2_ae_ccpvdz_cart.h5"), store_tuple=True
    )
    mo_leaf = MOs_data(num_mo=aos_data.num_ao, aos_data=aos_data, mo_coefficients=np.eye(aos_data.num_ao))
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=_make_j3(mo_leaf),
        jastrow_dtn_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


def _build_dense_agp_ham() -> Hamiltonian_data:
    """A dense-lambda AO-AGP geminal (``convert_from_MOs_to_AOs`` -- deliberately RANK-DEFICIENT)
    and no J3, so the c-axis exercises purely the dense AO-AGP path."""
    (structure_data, _aos, _mu, _md, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, "lithium.h5"), store_tuple=True
    )
    dense_agp = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=dense_agp)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


@pytest.mark.light
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("axis", ["R", "eps"])
def test_mo_backed_j3_pulay_now_analytic(quantity, axis):
    """MO-backed J3 on R / eps for a PLAIN (non dense-AO-AGP) MOLECULAR geminal: the analytic
    (folded bare-AO) path matches the ``_auto`` twin within the zone-derived tolerance. The
    MO->bare-AO J3 fold (``assemble._j3_mo_to_bare_ao``) is geminal-independent, so R/eps
    force/stress serve an MO-backed J3 on a plain MO/SD geminal, not only the dense-AO-AGP
    pairing the generated matrix exercises.
    """
    from jqmc._precision import get_tolerance_min

    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
        "strict",
    )
    ham = _build_mo_j3_ham()
    r_up, r_dn = _draw(ham.wavefunction_data.geminal_data)
    axis_flag = {"R": "opt_R", "eps": "opt_eps"}[axis]

    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {axis_flag: True}, analytic=True)[axis]
        b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {axis_flag: True}, analytic=False)[axis]
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), {axis_flag: True}, analytic=True)[axis]
        b = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), {axis_flag: True}, analytic=False)[axis]

    la_, lb_ = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la_) == len(lb_)
    for la, lb in zip(la_, lb_):
        np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


@pytest.mark.light
def test_mo_backed_j3_r_drift_matches_auto():
    """NARROW: the MO-J3 r-axis (drift) IS implemented for the MO leaf -- the analytic ln_Psi/r
    path runs and matches the ``_auto`` twin (the axis is not over-refused).
    """
    from jqmc._precision import get_tolerance_min

    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval"),
        "strict",
    )
    ham = _build_mo_j3_ham()
    r_up, r_dn = _draw(ham.wavefunction_data.geminal_data)
    a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_r": True}, analytic=True)["r"]
    b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_r": True}, analytic=False)["r"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


@pytest.mark.light
def test_dense_ao_agp_ln_psi_c_axis_now_analytic():
    """The potential-FREE ln_Psi c-axis (O_k) of a RANK-DEFICIENT dense AO-AGP WF is analytic and
    runs (verified vs ``_auto`` on the well-conditioned lift in the 8-set harness). Asserts the
    analytic path runs with no ``NotImplementedError``."""
    ham = _build_dense_agp_ham()
    r_up, r_dn = _draw(ham.wavefunction_data.geminal_data)
    derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_geminal": True}, analytic=True)


@pytest.mark.light
def test_dense_ao_agp_e_L_c_axis_now_analytic():
    """The e_L c-axis of a RANK-DEFICIENT dense AO-AGP WF is served analytically (dT/dc +
    dV_ecp/dc), a==b-verified vs the ``_auto`` twin on the well-conditioned lift in the 8-set
    harness. Here we assert the assembled path runs and returns a fully FINITE c-axis grad (no
    SVD-at-degeneracy NaN leak on this ``convert_from_MOs_to_AOs`` fixture, where ``jax.grad``
    through the natural-orbital SVD NaNs)."""
    ham = _build_dense_agp_ham()
    r_up, r_dn = _draw(ham.wavefunction_data.geminal_data)
    out = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), {"opt_geminal": True}, analytic=True)
    assert set(out) == {"c"}
    for leaf in jax.tree_util.tree_leaves(out["c"]):
        assert np.all(np.isfinite(np.asarray(leaf)))

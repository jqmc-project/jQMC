"""Native real-Gamma force / stress on the QE converter's dense-MO composite J3 (issue #104).

``jqmc-tool qe convert-to -j3 mo`` (``run_qe2jqmc(..., {"j3_basis_type": "mo"})``) on a k = 0 save
builds the J3 orbital as a DENSE real MO matrix (n_occ, n_ao) over the same AO basis as the
determinant. For ``kind="pgto+pw"`` (the converter default) and ``kind="pgto+bspline"`` that basis is
a composite ``AOs_composite_data`` (pGTO child plus a real plane-wave or B-spline child). Every
committed composite-J3 fixture wraps the composite in a SQUARE identity MO, and the real-Gamma
pGTO+B-spline fixtures carry no J3, so this shape was never exercised:

* the native force context (``lib.rs`` ``j3_pbc_ctx_owned``) and stress context
  (``j3_stress_ctx_owned``) lowered the MO J3 to AO rank (``M_ao = C^T M_mo C``) but handed the
  ``MOs_data`` wrapper, not its AO leaf, to ``pgto_j3_pieces_composite``, which accepts only an
  identity C: ``NotImplementedError: J3 force/stress with a non-identity MO-wrapped orbital is not
  ported`` from MCMC force / stress and GFMC force as soon as the j_matrix is nonzero;
* ``pgto_j3_pieces_composite`` knew only the plane-wave child: the B-spline child was taken as the
  pGTO child (``AttributeError: ... '_nucleus_index_jnp'``), and the real J3 force / stress kernels
  (``forces_pbc.rs``, ``stress_pbc.rs``) had no B-spline rows;
* the free bindings ``grad_T_pbc_wrt_R``, ``grad_T_pbc_wrt_strain`` and ``_grad_lnJ3_pbc_strain``
  passed the un-lowered orbital the same way.

Input: the committed QE save ``tests/qe_example_files/Li_gamma`` (Li, k = 0, ccECP auto-discovered),
converted by the public ``run_qe2jqmc`` with ``j1 = j2 = 1`` and ``j3_basis_type="mo"``. The
converter writes a zero ``j_matrix`` (the J3 is skipped), so the test replaces it by a seeded
symmetric one, as after a J3 optimization.

Reference, independent of the native code: reverse-mode AD of the XLA ``Re ln Psi``,
``Re compute_local_energy`` (identity quadrature rotation), ``compute_kinetic_energy`` and
``compute_Jastrow_three_body`` with respect to every ``Structure_data.positions`` leaf, the electron
coordinates, or a homogeneous strain of cell, nuclei and electrons
(``Hamiltonian_data._apply_strain`` and ``r @ F.T``), in the active precision policy. In full
precision the AD force on the pGTO+PW Hamiltonian agrees with a fourth-order central finite
difference of the same energy to 1.6e-9 (e_L) and 1.1e-11 (ln Psi) at h = 1e-3 (recorded in #104).
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

import functools
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_min, precision_policy_snapshot
from jqmc.atomic_orbital import AOs_bspline_data, AOs_composite_data, AOs_plane_wave_data
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import compute_Jastrow_three_body
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import compute_kinetic_energy, evaluate_ln_wavefunction
from tests._test_data_cache import qe_example_files_root

jax.config.update("jax_enable_x64", True)

# The tests read the committed QE save through an absolute path and write the converted
# Hamiltonian to a private temporary directory.
pytestmark = pytest.mark.with_chdir

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAVE = os.path.join(qe_example_files_root(), "Li_gamma", "out", "li.save")
_EXTRA_CHILD = {"pgto+pw": AOs_plane_wave_data, "pgto+bspline": AOs_bspline_data}
_J_SEED = 7
_J_SCALE = 0.3
_SEED = 1
_SCALE = 0.6
#: GFMC lattice spacing; the T-move measurement force / stress does not depend on it.
_ALAT = 0.3
_NN = 1
#: The converter's B-spline child is the small residual next to the pGTO block (|AO| <= 1.8e-3 on
#: this save), so its J3 rows sit near the comparison tolerance. The "amplified" cases multiply the
#: J3 copy of its spline coefficients by this factor (the determinant is unchanged), which makes the
#: B-spline rows of the J3 force / stress kernels of the same order as the pGTO rows.
_BSPLINE_AMPLIFICATION = 1000.0
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by the real J123 ln Psi / local energy and their derivatives.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "wf_kinetic",
    "ewald",
    "coulomb",
)


@functools.lru_cache(maxsize=None)
def _converted(kind):
    """Public QE conversion of the committed Li Gamma save with a J3 MO basis (cached per kind)."""
    from jqmc.jqmc_qe import run_qe2jqmc

    out = os.path.join(tempfile.mkdtemp(prefix="jqmc_j3mo_"), f"ham_{kind.replace('+', '_')}_j3mo.h5")
    run_qe2jqmc(
        {"qe_save_dir": _SAVE, "kind": kind, "upf_paths": [], "ecutwfc": None, "alpha_min": None},
        {"j1_parameter": 1.0, "j2_parameter": 1.0, "j3_basis_type": "mo"},
        out,
    )
    return out


def _converter_hamiltonian(kind, source_path=None):
    ham = Hamiltonian_data.load_from_hdf5(_converted(kind) if source_path is None else source_path)
    j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
    orb = j3.orb_data
    # The converter's J3: a DENSE real MO over a (pGTO, extra) composite.
    assert isinstance(orb, MOs_data), type(orb)
    assert isinstance(orb.aos_data, AOs_composite_data), type(orb.aos_data)
    gto, extra = orb.aos_data.children
    assert isinstance(extra, _EXTRA_CHILD[kind]), orb.aos_data.children
    coefficients = np.asarray(orb.mo_coefficients)
    assert not np.iscomplexobj(coefficients)
    assert coefficients.shape[0] < coefficients.shape[1] == int(gto.num_ao) + int(extra.num_ao), coefficients.shape
    return ham


def _with_seeded_j3(ham, order, extra="converter"):
    """Seeded symmetric j_matrix; ``extra_first`` swaps the J3 children and permutes the MO columns.

    ``extra="amplified"`` scales the J3 B-spline child by ``_BSPLINE_AMPLIFICATION``.
    """
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    shape = np.asarray(j3.j_matrix).shape
    n = shape[0]
    j_matrix = _J_SCALE * np.random.default_rng(_J_SEED).standard_normal(shape)
    j_matrix[:, :n] = 0.5 * (j_matrix[:, :n] + j_matrix[:, :n].T)
    orb = j3.orb_data
    if extra == "amplified":
        gto, bspline = orb.aos_data.children
        assert isinstance(bspline, AOs_bspline_data), orb.aos_data.children
        bspline = bspline.replace(spline_coefficients=np.asarray(bspline.spline_coefficients) * _BSPLINE_AMPLIFICATION)
        orb = orb.replace(aos_data=AOs_composite_data.build((gto, bspline)))
    else:
        assert extra == "converter", extra
    if order == "extra_first":
        gto, extra = orb.aos_data.children
        num_gto = int(gto.num_ao)
        c = np.asarray(orb.mo_coefficients)
        # The orbital (and so every observable) is unchanged: only the AO order differs.
        orb = orb.replace(
            aos_data=AOs_composite_data.build((extra, gto)),
            mo_coefficients=np.ascontiguousarray(np.concatenate([c[:, num_gto:], c[:, :num_gto]], axis=1)),
        )
    # NumPy arrays, as Hamiltonian_data.load_from_hdf5 stores them.
    j3 = j3.replace(orb_data=orb, j_matrix=np.asarray(j_matrix))
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jastrow.replace(jastrow_three_body_data=j3)))


def _without_j3(ham):
    wf = ham.wavefunction_data
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=None)))


def _configuration(ham):
    geminal = ham.wavefunction_data.geminal_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(_SEED)
    r_up = positions[np.arange(num_up) % len(positions)] + rng.normal(scale=_SCALE, size=(num_up, 3))
    r_dn = positions[np.arange(num_dn) % len(positions)] + rng.normal(scale=_SCALE, size=(num_dn, 3))
    return np.ascontiguousarray(r_up, dtype=np.float64), np.ascontiguousarray(r_dn, dtype=np.float64)


def _is_structure(x):
    return isinstance(x, Structure_data)


def _displaced(ham, d_positions):
    """Hamiltonian with every Structure_data.positions leaf displaced by ``d_positions``."""
    return jax.tree_util.tree_map(
        lambda x: x.replace(positions=jnp.asarray(x.positions) + d_positions) if _is_structure(x) else x,
        ham,
        is_leaf=_is_structure,
    )


def _ad_references(ham, r_up, r_dn):
    """Reverse-mode AD of Re ln Psi, Re e_L, T and J3 with respect to nuclei, electrons and strain.

    One Jacobian of the four scalars with respect to (nuclear displacement, strain, up electrons,
    down electrons) at zero displacement and zero strain: every partial is the derivative the
    native kernels return. The compiled programs are released afterwards (``jax.clear_caches``):
    the mixed-precision pGTO+B-spline programs map about 1e4 memory regions each, and keeping
    several alive exceeds the default ``vm.max_map_count`` (65530), which aborts the process.
    """
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))

    def scalars(d_positions, eps, a, b):
        deformation = jnp.eye(3) + eps
        h = _displaced(ham, d_positions)._apply_strain(deformation)
        a, b = a @ deformation.T, b @ deformation.T
        wf = h.wavefunction_data
        return jnp.stack(
            [
                jnp.real(evaluate_ln_wavefunction(wf, a, b)),
                jnp.real(compute_local_energy(h, a, b, jnp.eye(3))),
                jnp.real(compute_kinetic_energy(wf, a, b)),
                jnp.real(compute_Jastrow_three_body(wf.jastrow_data.jastrow_three_body_data, a, b)),
            ]
        )

    jac = jax.jacrev(scalars, argnums=(0, 1, 2, 3))(zero, jnp.zeros((3, 3)), up, dn)
    nuclei, strain, d_up, d_dn = (np.asarray(x, dtype=np.float64) for x in jac)
    jax.clear_caches()
    ln_psi, e_l, kinetic, j3_value = range(4)
    return {
        "pulay_nuclei": nuclei[ln_psi],
        "drift_up": d_up[ln_psi],
        "drift_dn": d_dn[ln_psi],
        "hf_nuclei": nuclei[e_l],
        "hf_up": d_up[e_l],
        "hf_dn": d_dn[e_l],
        "pulay_strain": strain[ln_psi],
        "hf_strain": strain[e_l],
        "kinetic_nuclei": nuclei[kinetic],
        "kinetic_strain": strain[kinetic],
        "j3_strain": strain[j3_value],
    }


def _hf_references_without_j3(ham, r_up, r_dn):
    """AD of Re e_L with the J3 removed: (d/dR, d/deps), for the J3-share (non-vacuity) check."""
    no_j3 = _without_j3(ham)
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)

    def e_l(d_positions, eps):
        deformation = jnp.eye(3) + eps
        h = _displaced(no_j3, d_positions)._apply_strain(deformation)
        return jnp.real(compute_local_energy(h, up @ deformation.T, dn @ deformation.T, jnp.eye(3)))

    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))
    nuclei, strain = (np.asarray(x, dtype=np.float64) for x in jax.grad(e_l, argnums=(0, 1))(zero, jnp.zeros((3, 3))))
    jax.clear_caches()
    return nuclei, strain


def _native_hf_measurement_outputs(ham, r_up, r_dn, driver):
    """The exact native HF measurement kernels used by the two public drivers.

    Inputs are the same unwrapped coordinates and identity quadrature rotation.
    Values are raw dE_L/dR and dE_L/deps (no Pulay or volume normalization).
    """
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    if driver == "MCMC":
        geminal_inv = np.asarray(
            _geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.float64
        )
        ctx = jqmc_rs.build_obs_ctx(
            geminal, jastrow, coulomb, False, True, True, True, precision_policy=precision_policy_snapshot()
        )
        force = np.asarray(ctx.grad_e_L_pbc_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)[0])[0]
        stress = np.asarray(ctx.grad_e_L_eps_pbc_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv))[0]
    else:
        assert driver == "GFMC_t", driver
        args = (geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False)
        force = np.asarray(jqmc_rs._gfmc_grad_e_L_ecp_pbc(*args)[0])
        stress = np.asarray(jqmc_rs._gfmc_stress_ecp_pbc_withj(*args))
    return {"force": force, "stress": stress}


@pytest.mark.parametrize(
    "kind, order, extra",
    [
        pytest.param("pgto+pw", "gto_first", "converter", marks=pytest.mark.heavy),
        pytest.param("pgto+pw", "extra_first", "converter", marks=pytest.mark.heavy),
        pytest.param("pgto+bspline", "gto_first", "converter", marks=pytest.mark.heavy),
        pytest.param("pgto+bspline", "extra_first", "converter", marks=pytest.mark.heavy),
        pytest.param("pgto+bspline", "gto_first", "amplified", marks=pytest.mark.heavy),
        pytest.param("pgto+bspline", "extra_first", "amplified", marks=pytest.mark.heavy),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_converter_dense_mo_composite_j3_real_forces_and_stress_match_ad(kind, order, extra):
    """Native real-Gamma forces and stress equal AD for the converter's dense-MO composite J3.

    The native force / stress context must serve a dense real MO over a pGTO(+)PW or
    pGTO(+)B-spline composite (no NotImplementedError for the dense MO, no AttributeError for the
    B-spline child). At one seeded configuration, with a seeded symmetric j_matrix and the J3 composite
    in its converted (pGTO, extra) order or in the (extra, pGTO) order with the MO columns permuted
    to match, and for pGTO+B-spline also with the J3 B-spline child amplified (``extra="amplified"``,
    so its rows are not a near-tolerance residual), every native quantity below must equal
    reverse-mode AD of the XLA energy in the same
    precision: the Pulay force and drift, the HF force on nuclei and electrons and the Pulay / HF
    stress of the MCMC driver context (``build_obs_ctx``) and of the free VMC kernels, the GFMC/LRDMC
    T-move measurement force and stress, the kinetic force (``grad_T_pbc_wrt_R``), the kinetic
    stress (``grad_T_pbc_wrt_strain``, pGTO+PW only: that free hook has no B-spline determinant
    rows) and the J3 value strain (``_grad_lnJ3_pbc_strain``). The J3 must change the HF force and
    the HF stress references by more than the comparison tolerance.
    """
    ham = _with_seeded_j3(_converter_hamiltonian(kind), order, extra)
    r_up, r_dn = _configuration(ham)
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ref = _ad_references(ham, r_up, r_dn)
    atol, rtol = get_tolerance_min(_ZONES, "medium")

    # Non-vacuous: the J3 changes the HF force and stress beyond the comparison tolerance.
    hf_nuclei_no_j3, hf_strain_no_j3 = _hf_references_without_j3(ham, r_up, r_dn)
    assert not np.allclose(ref["hf_nuclei"], hf_nuclei_no_j3, atol=atol, rtol=rtol), (
        "J3 share of the HF force is below tolerance"
    )
    assert not np.allclose(ref["hf_strain"], hf_strain_no_j3, atol=atol, rtol=rtol), (
        "J3 share of the HF stress is below tolerance"
    )

    geminal_inv = np.asarray(
        _geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.float64
    )
    ctx = jqmc_rs.build_obs_ctx(
        geminal, jastrow, coulomb, False, True, True, True, precision_policy=precision_policy_snapshot()
    )
    ctx_pulay = ctx.grad_ln_psi_pbc_batch(r_up[None], r_dn[None], geminal_inv)
    ctx_hf = ctx.grad_e_L_pbc_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    ctx_pulay_stress = ctx.grad_ln_psi_eps_pbc_batch(r_up[None], r_dn[None], geminal_inv)
    ctx_hf_stress = ctx.grad_e_L_eps_pbc_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    free_pulay = jqmc_rs.vmc_grad_ln_psi_pbc(geminal, jastrow, r_up, r_dn, geminal_inv[0])
    free_hf = jqmc_rs.vmc_grad_e_L_pbc(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, geminal_inv[0])
    free_pulay_stress = jqmc_rs.vmc_grad_ln_psi_eps_pbc(geminal, jastrow, r_up, r_dn, geminal_inv[0])
    free_hf_stress = jqmc_rs.vmc_grad_e_L_eps_pbc(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, geminal_inv[0])
    gfmc_args = (geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False)
    gfmc_hf = jqmc_rs._gfmc_grad_e_L_ecp_pbc(*gfmc_args)
    gfmc_stress = jqmc_rs._gfmc_stress_ecp_pbc_withj(*gfmc_args)
    kinetic_nuclei = jqmc_rs.grad_T_pbc_wrt_R(geminal, jastrow, r_up, r_dn)
    kinetic_strain = jqmc_rs.grad_T_pbc_wrt_strain(geminal, jastrow, r_up, r_dn)
    j3_strain = jqmc_rs._grad_lnJ3_pbc_strain(jastrow, r_up, r_dn)

    checks = [
        ("Pulay dR (MCMC driver ctx)", np.asarray(ctx_pulay[0])[0], ref["pulay_nuclei"]),
        ("drift dr_up (MCMC driver ctx)", np.asarray(ctx_pulay[1])[0], ref["drift_up"]),
        ("drift dr_dn (MCMC driver ctx)", np.asarray(ctx_pulay[2])[0], ref["drift_dn"]),
        ("HF dR (MCMC driver ctx)", np.asarray(ctx_hf[0])[0], ref["hf_nuclei"]),
        ("HF dr_up (MCMC driver ctx)", np.asarray(ctx_hf[1])[0], ref["hf_up"]),
        ("HF dr_dn (MCMC driver ctx)", np.asarray(ctx_hf[2])[0], ref["hf_dn"]),
        ("Pulay stress (MCMC driver ctx)", np.asarray(ctx_pulay_stress)[0], ref["pulay_strain"]),
        ("HF stress (MCMC driver ctx)", np.asarray(ctx_hf_stress)[0], ref["hf_strain"]),
        ("Pulay dR (VMC free kernel)", free_pulay[0], ref["pulay_nuclei"]),
        ("drift dr_up (VMC free kernel)", free_pulay[1], ref["drift_up"]),
        ("drift dr_dn (VMC free kernel)", free_pulay[2], ref["drift_dn"]),
        ("HF dR (VMC free kernel)", free_hf[0], ref["hf_nuclei"]),
        ("HF dr_up (VMC free kernel)", free_hf[1], ref["hf_up"]),
        ("HF dr_dn (VMC free kernel)", free_hf[2], ref["hf_dn"]),
        ("Pulay stress (VMC free kernel)", free_pulay_stress, ref["pulay_strain"]),
        ("HF stress (VMC free kernel)", free_hf_stress, ref["hf_strain"]),
        ("HF dR (GFMC T-move)", gfmc_hf[0], ref["hf_nuclei"]),
        ("HF dr_up (GFMC T-move)", gfmc_hf[1], ref["hf_up"]),
        ("HF dr_dn (GFMC T-move)", gfmc_hf[2], ref["hf_dn"]),
        ("HF stress (GFMC T-move)", gfmc_stress, ref["hf_strain"]),
        ("kinetic dT/dR (grad_T_pbc_wrt_R)", kinetic_nuclei, ref["kinetic_nuclei"]),
        ("J3 value strain (_grad_lnJ3_pbc_strain)", j3_strain, ref["j3_strain"]),
    ]
    if kind == "pgto+pw":
        # The free ``grad_T_pbc_wrt_strain`` hook has no B-spline DETERMINANT rows (independent of
        # the J3: it misses the same amount with the J3 removed), so the complete kinetic stress is
        # compared only for the pGTO+PW determinant. Its J3 rows run the same ``j3_pbc_kstress_w``
        # the driver HF stress above uses for both composites.
        checks.append(("kinetic dT/deps (grad_T_pbc_wrt_strain)", kinetic_strain, ref["kinetic_strain"]))
    for name, value, reference in checks:
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(reference)), (name, reference)
        assert np.all(np.isfinite(value)), (name, value)
        # Non-vacuous: the reference is distinguishable from zero at the comparison tolerance.
        assert np.max(np.abs(reference)) > atol + rtol * np.max(np.abs(reference)), (name, reference)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"Rust real-Gamma dense-MO composite J3 != AD reference: {name} ({kind}, J3 {order}, {extra})",
        )

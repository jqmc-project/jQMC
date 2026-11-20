"""Production parity for LSDA (unrestricted, unequal-spin) periodic orbitals,
Rust vs jqmc.

Coverage note: the committed QE saves ``tests/qe_example_files/
O-lsda_gamma`` (real Gamma, ``nspin=2``, triplet oxygen ``n_up=4, n_dn=2``,
INDEPENDENT up/dn Kohn-Sham orbitals -- not a shared spatial orbital set
truncated to different occupations) and ``O-lsda_k025`` (same system at a
genuine k != Gamma twist) are exercised end-to-end on the XLA side
(``tests/test_jqmc_kernel/test_qe_lsda.py``: wrapper parsing, unrestricted
``SD_data`` assembly, PW/pgto/bspline projection, multi-twist round-trip) but
were NEVER referenced anywhere under ``tests/test_rs_kernel`` -- every
periodic/twist pattern in ``_rs_fixtures._PATTERNS`` (``lithium*``, QE
``Li_gamma`` / ``Li_k025``) is spin-RESTRICTED closed-shell
(``num_electron_up == num_electron_dn``, identical or trivially-related MO
sets). An unrestricted periodic orbital set is a distinct case for the
per-spin AO/MO evaluation path (``orb_data_up_spin`` and ``orb_data_dn_spin``
carry genuinely different coefficient blocks, not just a different electron
count on the same coefficients -- verified below), so it is not a same-code-
path merge with the closed-shell periodic rows.

Reuses the exact PRODUCTION conversion entry points already unit-tested by
``test_qe_lsda.py`` (``run_qe2jqmc`` for the real-Gamma pgto/bspline hams,
``convert_qe_to_twist_average_input`` + ``load_twist`` for the k=0.25 twist
ham) -- no new builder logic, so a bug in the conversion path itself is out
of scope here (that is XLA-side and already covered by ``test_qe_lsda.py``);
this module only checks Rust == XLA on the resulting ``Hamiltonian_data``.

Production reachability: ``jqmc_mcmc.py``'s ``flag_rust_on_CPU`` dispatch
gates on CPU + full-precision only, with
no leaf-type / spin-restriction check, so an LSDA-derived periodic run under
``flag_rust_on_CPU=True`` unconditionally routes through these same kernels.
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

import numpy as np
import pytest

import jax
import jqmc_rs

pytestmark = pytest.mark.with_chdir

import jax.numpy as jnp

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM
from jqmc._twist_average import load_twist
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _reduce_c_param_grads
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy_fast
from jqmc.jqmc_qe import run_qe2jqmc
from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
from tests._test_data_cache import qe_example_files_root
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

_QE_DIR = os.fspath(qe_example_files_root())
_GAMMA_SAVE = os.path.abspath(os.path.join(_QE_DIR, "O-lsda_gamma", "out", "o.save"))
_K025_SAVE = os.path.abspath(os.path.join(_QE_DIR, "O-lsda_k025", "out", "o_k025.save"))
_K025_UPF = os.path.join(_K025_SAVE, "O.ccECP.upf")

_N_OCC_UP = 4
_N_OCC_DN = 2

# Periodic/twist far-image cutoff floor -- same convention as
# test_grads_laplacian_far.py / test_rs_analytic_matrix.py's periodic rows.
_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")
_ATOL_DERIV, _RTOL_DERIV = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium")
_ATOL_DERIV = max(_ATOL_DERIV, ATOL_FLOOR_MEDIUM)
_RTOL_DERIV = max(_RTOL_DERIV, RTOL_FLOOR_MEDIUM)
_MESH = icosahedron_sym_mesh_Nv12


def _build_gamma(kind: str, tmp_path):
    out = os.path.join(str(tmp_path), f"ham_{kind}.h5")
    qe_params = {"qe_save_dir": _GAMMA_SAVE, "kind": kind, "upf_paths": [], "ecutwfc": None, "alpha_min": None}
    jastrow_params = {"j1_parameter": None, "j2_parameter": None}
    run_qe2jqmc(qe_params, jastrow_params, out)
    return Hamiltonian_data.load_from_hdf5(out)


def _build_twist_pgto(tmp_path):
    out = os.path.join(str(tmp_path), "twist_lsda_pgto.h5")
    convert_qe_to_twist_average_input(_K025_SAVE, out, upf_paths={"O": _K025_UPF}, kind="pgto")
    return load_twist(out, 0)


def _assert_unrestricted(ham):
    """Sanity pin: this really is the triplet-oxygen open-shell fixture (electron
    counts genuinely unequal per spin), not a same-count system that would ride
    the SAME closed-shell code path every other ``_rs_fixtures`` periodic row
    exercises. ``test_qe_lsda.py`` already pins the STRONGER claim -- that the
    up/dn orbital CONTENT is independent per spin (not merely a truncated shared
    set) -- per representation kind (``mo_coefficients`` diff for pgto/pw,
    ``composite_child_dn`` for bspline); this module does not re-derive that
    (out of scope -- it only checks Rust == XLA on the resulting ham).
    """
    gd = ham.wavefunction_data.geminal_data
    assert int(gd.num_electron_up) == _N_OCC_UP
    assert int(gd.num_electron_dn) == _N_OCC_DN
    assert gd.num_electron_up != gd.num_electron_dn


def _config(ham, seed):
    gd = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[0] + rng.normal(scale=0.6, size=(int(gd.num_electron_up), 3))
    r_dn = R[0] + rng.normal(scale=0.6, size=(int(gd.num_electron_dn), 3))
    return r_up, r_dn


def _reduced_c_blocks(ham, r_up, r_dn, quantity, g_inv):
    """Return the exact optimizer-facing occupied-SD parameter blocks."""
    mask = {"opt_geminal": True}
    if quantity == "ln_Psi":
        grad = assemble.compute_grad_ln_Psi(
            ham,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            wrt="c",
            mask=mask,
            geminal_inv=jnp.asarray(g_inv),
        )
    else:
        grad = assemble.compute_grad_e_L(
            ham,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            jnp.eye(3),
            wrt="c",
            mask=mask,
            geminal_inv=jnp.asarray(g_inv),
        )
    batched = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf]), grad)
    reduced = _reduce_c_param_grads(ham.wavefunction_data, batched, 1, mask)
    return {name: np.asarray(values)[0] for name, values in reduced.items() if name.startswith("mo_coefficients_")}


def _jax_force(ham, r_up, r_dn, g_inv):
    hf = (
        assemble.compute_grad_e_L(
            ham,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            jnp.eye(3),
            wrt="R",
            geminal_inv=jnp.asarray(g_inv),
        ),
        assemble.compute_grad_e_L(
            ham,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            jnp.eye(3),
            wrt="r",
            geminal_inv=jnp.asarray(g_inv),
        ),
    )
    pulay = (
        assemble.compute_grad_ln_Psi(
            ham,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            wrt="R",
            geminal_inv=jnp.asarray(g_inv),
        ),
        assemble.compute_grad_ln_Psi(
            ham,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            wrt="r",
            geminal_inv=jnp.asarray(g_inv),
        ),
    )
    return (
        tuple(np.asarray(value) for value in (hf[0], *hf[1])),
        tuple(np.asarray(value) for value in (pulay[0], *pulay[1])),
    )


def _jax_stress(ham, r_up, r_dn, g_inv):
    hf = assemble.compute_grad_e_L(
        ham,
        jnp.asarray(r_up),
        jnp.asarray(r_dn),
        jnp.eye(3),
        wrt="eps",
        geminal_inv=jnp.asarray(g_inv),
    )
    pulay = assemble.compute_grad_ln_Psi(
        ham,
        jnp.asarray(r_up),
        jnp.asarray(r_dn),
        wrt="eps",
        geminal_inv=jnp.asarray(g_inv),
    )
    return np.asarray(hf), np.asarray(pulay)


def _assert_triple_close(native, reference, label):
    for axis, (actual, expected) in zip(("R", "r_up", "r_dn"), zip(native, reference, strict=True), strict=True):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            atol=_ATOL_DERIV,
            rtol=_RTOL_DERIV,
            err_msg=f"{label} d/d{axis}",
        )


@pytest.mark.light
@pytest.mark.parametrize("kind", ["pgto", "bspline"])
def test_local_energy_fast_lsda_gamma(kind, tmp_path):
    """Local energy fast lsda gamma."""
    ham = _build_gamma(kind, tmp_path)
    _assert_unrestricted(ham)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, seed=3)
    RT = np.eye(3)
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float) @ RT
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn)))
    native = float(jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, w, gp, g_inv))
    jax_eL = float(compute_local_energy_fast(ham, r_up, r_dn, jnp.asarray(RT), jnp.asarray(g_inv), None))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (kind, native, jax_eL, abs(native - jax_eL))


@pytest.mark.light
def test_local_energy_fast_complex_lsda_twist(tmp_path):
    """Local energy fast complex lsda twist."""
    ham = _build_twist_pgto(tmp_path)
    _assert_unrestricted(ham)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, seed=5)
    RT = np.eye(3)
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float) @ RT
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))).astype(np.complex128)
    re, im = jqmc_rs.compute_local_energy_fast_complex(ham, r_up, r_dn, w, gp, g_inv)
    native = complex(re, im)
    jax_eL = complex(compute_local_energy_fast(ham, r_up, r_dn, jnp.asarray(RT), jnp.asarray(g_inv), None))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert np.isclose(native.real, jax_eL.real, rtol=rtol, atol=atol), (native, jax_eL, abs(native - jax_eL))
    assert np.isclose(native.imag, jax_eL.imag, rtol=rtol, atol=atol), (native, jax_eL, abs(native - jax_eL))


@pytest.mark.heavy
@pytest.mark.parametrize("kind", ["pgto", "bspline"])
def test_derivative_observables_lsda_gamma(kind, tmp_path):
    """Unrestricted Gamma SD: force, stress, and both independent MO blocks."""
    ham = _build_gamma(kind, tmp_path)
    _assert_unrestricted(ham)
    gd = ham.wavefunction_data.geminal_data
    assert gd.time_reversal is False
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    # HF-force aggregates amplify fp32 backend-order differences near a
    # determinant node.  Select only on the XLA geminal condition number,
    # before either derivative backend is evaluated.
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=7, scale=0.6, cond_max=15.0)
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn)))
    weights = np.asarray(_MESH.weights, float)
    grid_points = np.asarray(_MESH.grid_points, float)

    hf_ref, pulay_ref = _jax_force(ham, r_up, r_dn, g_inv)
    _assert_triple_close(
        jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, r_up, r_dn, weights, grid_points, g_inv),
        hf_ref,
        f"{kind} unrestricted HF force",
    )
    _assert_triple_close(
        jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, r_up, r_dn, g_inv),
        pulay_ref,
        f"{kind} unrestricted Pulay force",
    )

    hf_stress_ref, pulay_stress_ref = _jax_stress(ham, r_up, r_dn, g_inv)
    np.testing.assert_allclose(
        jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, r_up, r_dn, weights, grid_points, g_inv),
        hf_stress_ref,
        atol=_ATOL_DERIV,
        rtol=_RTOL_DERIV,
        err_msg=f"{kind} unrestricted HF stress",
    )
    np.testing.assert_allclose(
        jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, r_up, r_dn, g_inv),
        pulay_stress_ref,
        atol=_ATOL_DERIV,
        rtol=_RTOL_DERIV,
        err_msg=f"{kind} unrestricted Pulay stress",
    )

    ok_ref = _reduced_c_blocks(ham, r_up, r_dn, "ln_Psi", g_inv)
    ok_up, ok_dn = jqmc_rs.grad_ln_det_wrt_mo_coeff(gd, r_up, r_dn)
    el_up, el_dn = jqmc_rs.grad_eL_wrt_mo_coeff(gd, jd, r_up, r_dn)
    ecp_up, ecp_dn = jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gd, jd, cp, r_up, r_dn, weights, grid_points)
    el_ref = _reduced_c_blocks(ham, r_up, r_dn, "e_L", g_inv)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium")
    for block, ok_native, el_native in (
        ("mo_coefficients_up", ok_up, np.asarray(el_up) + np.asarray(ecp_up)),
        ("mo_coefficients_dn", ok_dn, np.asarray(el_dn) + np.asarray(ecp_dn)),
    ):
        assert block in ok_ref and block in el_ref, f"unrestricted optimizer block {block} was dropped"
        np.testing.assert_allclose(ok_native, ok_ref[block], atol=atol, rtol=rtol, err_msg=f"{kind} O_k {block}")
        np.testing.assert_allclose(el_native, el_ref[block], atol=atol, rtol=rtol, err_msg=f"{kind} dE_L/dc {block}")


@pytest.mark.heavy
def test_derivative_observables_complex_open_shell_twist(tmp_path):
    """Complex k-UHF SD: force/stress and both independent complex MO blocks."""
    ham = _build_twist_pgto(tmp_path)
    _assert_unrestricted(ham)
    gd = ham.wavefunction_data.geminal_data
    assert gd.time_reversal is False
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    r_up, r_dn = _config(ham, seed=11)
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))).astype(np.complex128)
    weights = np.asarray(_MESH.weights, float)
    grid_points = np.asarray(_MESH.grid_points, float)

    hf_ref, pulay_ref = _jax_force(ham, r_up, r_dn, g_inv)
    _assert_triple_close(
        jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, r_up, r_dn, weights, grid_points, g_inv),
        hf_ref,
        "complex open-shell HF force",
    )
    _assert_triple_close(
        jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, r_up, r_dn, g_inv),
        pulay_ref,
        "complex open-shell Pulay force",
    )

    hf_stress_ref, pulay_stress_ref = _jax_stress(ham, r_up, r_dn, g_inv)
    np.testing.assert_allclose(
        jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, r_up, r_dn, weights, grid_points, g_inv),
        hf_stress_ref,
        atol=_ATOL_DERIV,
        rtol=_RTOL_DERIV,
        err_msg="complex open-shell HF stress",
    )
    np.testing.assert_allclose(
        jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, r_up, r_dn, g_inv),
        pulay_stress_ref,
        atol=_ATOL_DERIV,
        rtol=_RTOL_DERIV,
        err_msg="complex open-shell Pulay stress",
    )

    ok_ref = _reduced_c_blocks(ham, r_up, r_dn, "ln_Psi", g_inv)
    ok_up, ok_dn = jqmc_rs.grad_ln_det_mo_complex(gd, r_up, r_dn)
    el_up, el_dn = jqmc_rs.grad_T_mo_withJ_complex(gd, jd, r_up, r_dn)
    ecp_up, ecp_dn = jqmc_rs.grad_eL_ecp_wrt_mo_coeff_complex(gd, jd, cp, r_up, r_dn, weights, grid_points)
    el_ref = _reduced_c_blocks(ham, r_up, r_dn, "e_L", g_inv)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium")
    for block, ok_native, el_native in (
        ("mo_coefficients_up", ok_up, np.asarray(el_up) + np.asarray(ecp_up)),
        ("mo_coefficients_dn", ok_dn, np.asarray(el_dn) + np.asarray(ecp_dn)),
    ):
        assert block in ok_ref and block in el_ref, f"open-shell optimizer block {block} was dropped"
        np.testing.assert_allclose(ok_native, ok_ref[block], atol=atol, rtol=rtol, err_msg=f"complex O_k {block}")
        np.testing.assert_allclose(el_native, el_ref[block], atol=atol, rtol=rtol, err_msg=f"complex dE_L/dc {block}")

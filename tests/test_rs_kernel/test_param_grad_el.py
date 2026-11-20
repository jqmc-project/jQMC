"""Layer-1 equivalence: d(e_L)/dc = d(T)/dc (Rust) vs jax.grad(compute_kinetic_energy)
-- the production `_auto` oracle. Covers geminal lambda / mo_coefficients (jnone +
withJ), Jastrow J1/J2 param, J3 j_matrix. Real /
Gamma, molecular (CART + SPHE).

Hamiltonians come from the committed-fixture builders
(``tests.test_rs_kernel._rs_fixtures.rs_ham``), J3 rows are selected by an explicit
``is_j3`` parametrize flag, and tolerances come from the dedicated XLA--Rust
consistency helper.

Local pattern extensions (not yet named in the committed ``_rs_fixtures._PATTERNS``
registry) are built directly via ``_build_ham_for_cell`` -- the SAME shared builder
``rs_ham`` wraps, imported read-only from ``_generated_matrix_base``
-- covering: dense-AO-AGP + ECP + sphe (well-conditioned by AO family alone, see
``_agp_dense_constructible``), dense-AO-AGP + active J3 on cart AE (a plain
``with_jastrow=True`` AGP build), and SD/AGP + ECP + sphe + active J3/DTN (J3 kind
is AO-family-keyed -- ``sphe_gto`` resolves to the same ``"pgto"`` leaf as
``cart_gto``; DTN only needs ``structure_data``, so it's AO-family independent).
All three were verified buildable before use (see the sibling files'
docstrings for the same rationale).
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

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._precision import get_tolerance_xla_rust_consistency as _backend_consistency_tolerance
from jqmc._setting import RTOL_RS_ECP_PARAMETER_SM_REBUILD, RTOL_RS_ECP_PARAMETER_SM_REBUILD_ILL_CONDITIONED
from jqmc.derivatives.kinetic import grad_T_wrt_mo_withJ  # noqa: E402
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from jqmc.wavefunction import Wavefunction_data  # noqa: E402
from jqmc.wavefunction import compute_kinetic_energy as CK

get_tolerance_xla_rust_consistency = partial(_backend_consistency_tolerance, level="medium")

from tests._ecp_high_l import assert_fixture_has_live_high_l_channel
from tests._ecp_min_image_reference import min_image_distances
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg, rs_ham

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

_LOCAL_PATTERNS = {
    "mol_ecp_sphe_agp_jnone": ("water_ccecp_ccpvqz.h5", False, None, "AGP"),
    "mol_ecp_sphe_agp_j123": ("water_ccecp_ccpvqz.h5", True, {"j3": "pgto"}, "AGP"),
    "mol_cart_agp_j12": ("H2_ae_ccpvdz_cart.h5", True, None, "AGP"),
    "mol_cl2_ecp_cart_j3dtn": ("Cl2_ecp_ccpvtz_cart.h5", True, {"j3": "cart_gto"}, "SD"),
    "mol_ecp_sphe_j123": ("water_ccecp_ccpvqz.h5", True, {"j3": "pgto"}, "SD"),
    "mol_ecp_sphe_jdtn": ("water_ccecp_ccpvqz.h5", True, {"dtn": True}, "SD"),
    # Periodic real-Gamma AGPn (natural-orbital diagonal form) for the lambda_diag map.
    "pbc_agpn_sphepgto": ("lithium.h5", False, None, "AGPn"),
    "mol_ecp_sphe_j3dtn": ("water_ccecp_ccpvqz.h5", True, {"j3": "sphe_gto", "dtn": True}, "SD"),
    "pbc_gamma_sphepgto_j3dtn": ("lithium.h5", True, {"j3": "pgto", "dtn": True}, "SD"),
    "twist_agp_jnone": ("lithium_k025.h5", False, None, "AGP"),
}


def _ham(pattern):
    if pattern in _LOCAL_PATTERNS:
        return _build_ham_for_cell(*_LOCAL_PATTERNS[pattern])
    return rs_ham(pattern)


def _config(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[np.arange(nu) % len(R)] + rng.normal(scale=1.0, size=(nu, 3))
    r_dn = R[np.arange(nd) % len(R)] + rng.normal(scale=1.0, size=(nd, 3))
    return r_up, r_dn


def _close(a, b, atol, rtol):
    """PER-ELEMENT parity check; raises with the offending indices.

    A GLOBAL max-norm check -- ``max|a-b| <= atol or max|a-b| <= rtol * max|b|`` -- would
    set the tolerance band of every element by the LARGEST element of the reference, so a
    small component carrying a 100% relative error would pass as long as some other
    component is orders of magnitude bigger.

    This delegates to ``np.testing.assert_allclose``: each element is judged against ITS
    OWN magnitude (``|a-b| <= atol + rtol*|b|``), a shape mismatch fails, NaN on either
    side fails (the suite-wide tests/_nan_guard.py guard removes the ``equal_nan=True`` default),
    and the failure report names the mismatching indices instead of one global number.
    Returns True so the existing ``assert _close(...)`` call sites keep their context
    message; a violation raises from inside.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 and b.size == 0:
        return True
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    return True


def _rand_j3(ham, seed=7):
    """Inject a numpy (non-zero) j_matrix; committed j123 fixtures ship zeros."""
    wf = ham.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    n = np.asarray(j3.j_matrix).shape[0]
    jm = np.random.default_rng(seed).normal(scale=0.2, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    jd = wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm))
    return jd


_KINETIC_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap")


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern,is_j3",
    [
        ("mol_cart_agp_jnone", False),
        ("mol_ecp_sphe_agp_jnone", False),
        ("mol_cart_agp_j12", False),
        ("mol_ecp_sphe_agp_j123", True),
    ],
)
def test_grad_el_lambda(pattern, is_j3):
    """Grad el lambda."""
    ham = _ham(pattern)
    jd = _rand_j3(ham) if is_j3 else ham.wavefunction_data.jastrow_data
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    ana = jqmc_rs.grad_eL_wrt_lambda(gem, jd, ru, rd)
    ref = jax.grad(lambda g: CK(Wavefunction_data(jastrow_data=jd, geminal_data=g), ru, rd))(gem).lambda_matrix
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern,is_j3",
    [
        ("mol_cart_sd_jnone", False),
        ("mol_ecp_sphe_jnone", False),
        ("mol_cart_sd_j12", False),
        ("mol_ecp_sphe_j123", True),
    ],
)
def test_grad_el_mo(pattern, is_j3):
    """Grad el MO."""
    ham = _ham(pattern)
    jd = _rand_j3(ham) if is_j3 else ham.wavefunction_data.jastrow_data
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    au, ad = jqmc_rs.grad_eL_wrt_mo_coeff(gem, jd, ru, rd)
    g = jax.grad(lambda G: CK(Wavefunction_data(jastrow_data=jd, geminal_data=G), ru, rd))(gem)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(au), np.asarray(g.orb_data_up_spin.mo_coefficients)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    comparison_actual_2, comparison_reference_2 = np.asarray(ad), np.asarray(g.orb_data_dn_spin.mo_coefficients)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
def test_grad_el_mo_consumes_supplied_geminal_inverse():
    """The optional inverse is data, not a signature-only production argument."""
    ham = _ham("mol_cart_sd_jdtn")
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    # A fixed point with cond(G) ~= 1.4e7 would let the deliberately perturbed
    # inverse amplify backend rounding instead of testing whether the supplied
    # inverse was consumed.  Acceptance depends only on XLA G.
    ru, rd = draw_well_conditioned_cfg(ham, seed=13, scale=1.0, cond_max=700.0)
    g = np.asarray(compute_geminal_all_elements(gem, jnp.asarray(ru), jnp.asarray(rd)))
    supplied = np.linalg.inv(g)
    supplied[0, 0] += 0.01

    fresh = tuple(np.asarray(x) for x in jqmc_rs.grad_eL_wrt_mo_coeff(gem, jd, ru, rd))
    native = tuple(np.asarray(x) for x in jqmc_rs.grad_eL_wrt_mo_coeff(gem, jd, ru, rd, supplied))
    reference = tuple(
        np.asarray(x)
        for x in grad_T_wrt_mo_withJ(
            gem,
            jd,
            jnp.asarray(ru),
            jnp.asarray(rd),
            geminal_inv=jnp.asarray(supplied),
        )
    )
    assert any(not np.array_equal(with_inv, without_inv) for with_inv, without_inv in zip(native, fresh, strict=True))
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    for spin, with_inv, expected in zip(("up", "dn"), native, reference, strict=True):
        np.testing.assert_allclose(
            with_inv,
            expected,
            atol=atol,
            rtol=rtol,
            err_msg=f"supplied geminal inverse was not consumed on the {spin} MO block",
        )


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agpn_jnone", "pbc_agpn_sphepgto"])
def test_grad_el_lambda_diag(pattern):
    """AGPn dE_L on the natural-orbital leaves: ``grad_eL_wrt_lambda_diag`` (paired-block
    diagonal + unpaired columns of the dense-``lambda`` kinetic gradient) equals
    ``jax.grad(compute_kinetic_energy)`` on ``lambda_diag`` / ``lambda_unpaired``."""
    ham = _ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    d, u = jqmc_rs.grad_eL_wrt_lambda_diag(gem, jd, ru, rd)
    g = jax.grad(lambda G: CK(Wavefunction_data(jastrow_data=jd, geminal_data=G), ru, rd))(gem)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(d), np.asarray(g.lambda_diag)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    if np.asarray(gem.lambda_unpaired).size:
        comparison_actual_2, comparison_reference_2 = np.asarray(u), np.asarray(g.lambda_unpaired)
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
def test_grad_el_lambda_diag_non_full_rank_does_not_take_structural_zero_branch():
    """A virtual-occupation AGPn must execute the reverse pass, not return zero."""
    from jqmc.determinant import AGPn_data, _smetric_polar_retract
    from jqmc.molecular_orbital import MOs_data

    ham = _ham("mol_h2_cart_sd_jnone")
    sd = ham.wavefunction_data.geminal_data
    occupied = np.asarray(sd.orb_data_up_spin.mo_coefficients)
    for basis_row in np.eye(occupied.shape[1]):
        seed = np.vstack([occupied, basis_row])
        if np.linalg.matrix_rank(seed) == seed.shape[0]:
            break
    else:
        raise AssertionError("could not construct a linearly independent virtual test orbital")
    coefficients = _smetric_polar_retract(seed, sd.orb_data_up_spin.aos_data)
    mos_up = MOs_data(
        num_mo=coefficients.shape[0],
        aos_data=sd.orb_data_up_spin.aos_data,
        mo_coefficients=np.ascontiguousarray(coefficients),
    )
    gem = AGPn_data.build(
        num_electron_up=int(sd.num_electron_up),
        num_electron_dn=int(sd.num_electron_dn),
        mos_up=mos_up,
        time_reversal=True,
    )
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _config(ham)
    assert gem.lambda_unpaired.size == 0
    assert gem.orb_num_up > len(ru) and gem.orb_num_dn > len(rd)

    diag_native, unpaired_native = jqmc_rs.grad_eL_wrt_lambda_diag(gem, jd, ru, rd)
    grad_ref = jax.grad(lambda value: CK(Wavefunction_data(jastrow_data=jd, geminal_data=value), ru, rd))(gem)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    assert np.any(np.abs(np.asarray(diag_native)) > atol), "false-predicate branch returned the structural zero"
    comparison_actual_1, comparison_reference_1 = np.asarray(diag_native), np.asarray(grad_ref.lambda_diag)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    assert np.asarray(unpaired_native).size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["pw_gamma_jnone", "pbc_composite_jnone"])
def test_grad_el_mo_pw_composite(pattern):
    """SD dE_L (= dT/dC) on the mo_coefficients for the pure plane-wave and pGTO(+)PW
    composite leaves: ``grad_eL_wrt_mo_coeff`` contracts the FULL AO stack (gto + PW
    child) via ``aos_stack_value_grad_lap``, matching ``jax.grad(compute_kinetic_energy)``."""
    ham = _ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    au, ad = jqmc_rs.grad_eL_wrt_mo_coeff(gem, jd, ru, rd)
    g = jax.grad(lambda G: CK(Wavefunction_data(jastrow_data=jd, geminal_data=G), ru, rd))(gem)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(au), np.asarray(g.orb_data_up_spin.mo_coefficients)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    comparison_actual_2, comparison_reference_2 = np.asarray(ad), np.asarray(g.orb_data_dn_spin.mo_coefficients)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j12", "mol_ecp_sphe_j12"])
def test_grad_el_jastrow_12(pattern):
    """Grad el jastrow 12."""
    ham = _ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    a1 = jqmc_rs.grad_eL_wrt_jastrow_1b_param(gem, jd, ru, rd)
    r1 = jax.grad(lambda d: CK(Wavefunction_data(jastrow_data=d, geminal_data=gem), ru, rd))(
        jd
    ).jastrow_one_body_data.jastrow_1b_param
    comparison_actual_1, comparison_reference_1 = np.asarray(a1), np.asarray(r1)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    a2 = jqmc_rs.grad_eL_wrt_jastrow_2b_param(gem, jd, ru, rd)
    r2 = jax.grad(lambda d: CK(Wavefunction_data(jastrow_data=d, geminal_data=gem), ru, rd))(
        jd
    ).jastrow_two_body_data.jastrow_2b_param
    comparison_actual_2, comparison_reference_2 = np.asarray(a2), np.asarray(r2)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j123", "mol_ecp_sphe_j123"])
def test_grad_el_j3(pattern):
    """Grad el j3."""
    ham = _ham(pattern)
    jd = _rand_j3(ham)
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    ana = jqmc_rs.grad_eL_wrt_jastrow_3b_jmatrix(gem, jd, ru, rd)
    ref = jax.grad(lambda d: CK(Wavefunction_data(jastrow_data=d, geminal_data=gem), ru, rd))(
        jd
    ).jastrow_three_body_data.j_matrix
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


# ---- ECP non-local parameter gradient (d(V_ecp)/dc; completes d(e_L)/dc for ECP) ----


def _one_move_mesh():
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    w = np.array(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.array(icosahedron_sym_mesh_Nv12.grid_points, float)  # RT = eye
    return w, gp


_ECP_ZONES = ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval")


@pytest.mark.light
def test_grad_ecp_lambda():
    """Grad ECP lambda."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

    ham = _ham("mol_ecp_sphe_agp_jnone")
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    w, gp = _one_move_mesh()
    RT = np.eye(3)
    ru, rd = _config(ham)
    ana = jqmc_rs.grad_eL_ecp_wrt_lambda(gem, jd, cd, ru, rd, w, gp)
    ref = jax.grad(
        lambda lam: compute_ecp_non_local_parts_nearest_neighbors(
            cd, Wavefunction_data(jastrow_data=jd, geminal_data=gem.replace(lambda_matrix=lam)), ru, rd, RT
        )[3]
    )(np.asarray(gem.lambda_matrix))
    atol, rtol = _backend_consistency_tolerance(_ECP_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


@pytest.mark.light
def test_grad_ecp_mo():
    """Grad ECP MO."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

    ham = _ham("mol_ecp_sphe_jnone")
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    w, gp = _one_move_mesh()
    RT = np.eye(3)
    ru, rd = _config(ham)
    au, ad = jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gem, jd, cd, ru, rd, w, gp)

    def fwd(Cu, Cd):
        ou = gem.orb_data_up_spin.replace(mo_coefficients=Cu)
        od = gem.orb_data_dn_spin.replace(mo_coefficients=Cd)
        g2 = gem.replace(orb_data_up_spin=ou, orb_data_dn_spin=od)
        return compute_ecp_non_local_parts_nearest_neighbors(
            cd, Wavefunction_data(jastrow_data=jd, geminal_data=g2), ru, rd, RT
        )[3]

    ru_, rd_ = jax.grad(fwd, argnums=(0, 1))(
        np.asarray(gem.orb_data_up_spin.mo_coefficients), np.asarray(gem.orb_data_dn_spin.mo_coefficients)
    )
    atol, rtol = _backend_consistency_tolerance(_ECP_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(au), np.asarray(ru_)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    comparison_actual_2, comparison_reference_2 = np.asarray(ad), np.asarray(rd_)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


# ---- A/B parity: SM-incremental O_k reduce vs the fresh per-point rebuild ----
# The ECP-nl mo/lambda O_k reduce (``grad_ecp_{mo,lambda}``) replaced the per-mesh-
# point geminal rebuild with a rank-1 Sherman-Morrison increment off the SINGLE base
# ``G^{-1}`` (reused per candidate -- never a fresh per-point LU, so near-singular
# candidates stay finite, unlike the fresh-LU NaN mode). ``JQMC_ECP_OK_FORCE_REBUILD``
# forces EVERY candidate down the fresh per-point rebuild (the pre-optimization
# reference), so setting it recovers the exact per-point path for an A/B comparison.
# A near-singular candidate is seeded by placing two same-spin electrons ~1e-7 apart
# (near-degenerate geminal rows -> tiny det ratios -> the 1/d division is stressed);
# both paths must stay finite and agree at fp-association level.


# ``jqmc_rs.route_predicate_counters()`` indices of ``param_grad_ecp::sm_ratio_ok``
# mesh candidates that took the SM increment, and those
# whose ``sm_ratio_ok`` declined to the fresh per-point rebuild.
_SM_RATIO_ENGAGED = 8
_SM_RATIO_DECLINED = 9


def _sm_ratio_routes(call):
    """Run ``call`` in a fresh counter window; return (result, SM increments, rebuild declines)."""
    jqmc_rs.gfmc_route_counters_reset()
    out = call()
    counters = list(jqmc_rs.route_predicate_counters())
    return out, counters[_SM_RATIO_ENGAGED], counters[_SM_RATIO_DECLINED]


def _near_singular_config(ham, seed=3):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[np.arange(nu) % len(R)] + rng.normal(scale=0.3, size=(nu, 3))
    r_dn = R[np.arange(nd) % len(R)] + rng.normal(scale=0.3, size=(nd, 3))
    if nu >= 2:
        r_up[1] = r_up[0] + 1e-7  # near-coincident same-spin pair => near-singular geminal
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("singular", [False, True])
def test_grad_ecp_mo_sm_vs_rebuild_ab(singular, monkeypatch):
    """Grad ECP MO sm vs rebuild ab."""
    ham = _ham("mol_ecp_sphe_jnone")
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    w, gp = _one_move_mesh()
    ru, rd = _near_singular_config(ham) if singular else _config(ham)
    monkeypatch.delenv("JQMC_ECP_OK_FORCE_REBUILD", raising=False)
    (fu, fd), n_sm, n_rb = _sm_ratio_routes(lambda: jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gem, jd, cd, ru, rd, w, gp))
    fu, fd = np.asarray(fu), np.asarray(fd)
    monkeypatch.setenv("JQMC_ECP_OK_FORCE_REBUILD", "1")
    (bu, bd), forced_sm, forced_rb = _sm_ratio_routes(lambda: jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gem, jd, cd, ru, rd, w, gp))
    bu, bd = np.asarray(bu), np.asarray(bd)
    # Route witness: the forced reference declines every mesh candidate; the fast call
    # makes the same number of sm_ratio_ok decisions, and at the well-conditioned
    # config every one of them takes the SM increment (the decline does not fire).
    assert forced_sm == 0 and forced_rb > 0, (forced_sm, forced_rb)
    assert n_sm + n_rb == forced_rb, (n_sm, n_rb, forced_rb)
    if not singular:
        assert (n_sm, n_rb) == (forced_rb, 0), (n_sm, n_rb)
    # Robustness headline: reusing the base G^{-1} (never a fresh per-candidate LU)
    # keeps BOTH paths finite even for a near-degenerate geminal.
    assert np.isfinite(fu).all() and np.isfinite(fd).all()
    assert np.isfinite(bu).all() and np.isfinite(bd).all()
    # Both paths compute Sum_mp v_grid (O_k(r'_mp) - O_k_base). Well-conditioned:
    # rank-1 SM vs fresh-LU differ only by fp association (~1e-11 rel). Near-singular:
    # the tiny det ratios amplify fp noise, so they agree only to the conditioning
    # floor (~1e-7 rel) -- the ill-conditioned reduce itself carries no more digits in
    # EITHER path. (Candidates with |d| < sqrt(eps) take the shared rebuild branch, so
    # this env-forced all-rebuild reference also validates that fallback wiring.)
    rel = RTOL_RS_ECP_PARAMETER_SM_REBUILD_ILL_CONDITIONED if singular else RTOL_RS_ECP_PARAMETER_SM_REBUILD
    scale = max(np.abs(bu).max(), np.abs(bd).max(), 1.0)
    du = np.abs(fu - bu).max()
    dd = np.abs(fd - bd).max()
    assert du <= rel * scale, (du, scale)
    assert dd <= rel * scale, (dd, scale)


@pytest.mark.light
@pytest.mark.parametrize("seed", list(range(1, 13)))
def test_grad_ecp_mo_sm_vs_rebuild_near_coincident(seed, monkeypatch):
    """dC SM reduce stays on the rebuild at every near-coincident base.

    A near-coincident same-spin pair makes the BASE geminal near-singular, and the
    mesh moves of that pair lift it (det ratio ``|d|`` ~ 1e3..1e8). The SM increment is
    built from the base inverse, so its error grows like ``eps * kappa(G) * |d|``;
    before the conditioning-aware upper guard the SM result was 3e-5..4e-4 (full, 10 of
    these 12 configurations) and 3e-6..9e-4 (mixed, 11 of 12) relative off the per-point
    rebuild (seed 3, the single config of ``test_grad_ecp_mo_sm_vs_rebuild_ab[True]``,
    passed by chance until an fp32 AO rounding change moved it in mixed). Seed 7 has a moderately
    conditioned base (kappa ~ 6e6) with a ``|d|`` ~ 4e3 candidate: a ``|d|``-only
    threshold misses it, the ``kappa * |d|`` guard catches it.
    """
    ham = _ham("mol_ecp_sphe_jnone")
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    w, gp = _one_move_mesh()
    ru, rd = _near_singular_config(ham, seed=seed)
    monkeypatch.delenv("JQMC_ECP_OK_FORCE_REBUILD", raising=False)
    fast, n_sm, n_rb = _sm_ratio_routes(lambda: jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gem, jd, cd, ru, rd, w, gp))
    fu, fd = (np.asarray(x) for x in fast)
    monkeypatch.setenv("JQMC_ECP_OK_FORCE_REBUILD", "1")
    ref, forced_sm, forced_rb = _sm_ratio_routes(lambda: jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gem, jd, cd, ru, rd, w, gp))
    bu, bd = (np.asarray(x) for x in ref)
    # Route witness: at every near-coincident base the sm_ratio_ok decline FIRES for
    # some mesh candidates (they take the fresh per-point rebuild instead of the SM
    # increment); the decision count equals the forced all-rebuild reference's.
    assert forced_sm == 0 and forced_rb > 0, (forced_sm, forced_rb)
    assert n_sm + n_rb == forced_rb, (n_sm, n_rb, forced_rb)
    assert n_rb > 0, (n_sm, n_rb)
    assert np.isfinite(fu).all() and np.isfinite(fd).all()
    assert np.isfinite(bu).all() and np.isfinite(bd).all()
    scale = max(np.abs(bu).max(), np.abs(bd).max(), 1.0)
    du = np.abs(fu - bu).max()
    dd = np.abs(fd - bd).max()
    assert du <= RTOL_RS_ECP_PARAMETER_SM_REBUILD_ILL_CONDITIONED * scale, (du, scale)
    assert dd <= RTOL_RS_ECP_PARAMETER_SM_REBUILD_ILL_CONDITIONED * scale, (dd, scale)


@pytest.mark.light
@pytest.mark.parametrize("singular", [False, True])
def test_grad_ecp_lambda_sm_vs_rebuild_ab(singular, monkeypatch):
    """Grad ECP lambda sm vs rebuild ab."""
    ham = _ham("mol_ecp_sphe_agp_jnone")
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    w, gp = _one_move_mesh()
    ru, rd = _near_singular_config(ham) if singular else _config(ham)
    monkeypatch.delenv("JQMC_ECP_OK_FORCE_REBUILD", raising=False)
    fast, n_sm, n_rb = _sm_ratio_routes(lambda: jqmc_rs.grad_eL_ecp_wrt_lambda(gem, jd, cd, ru, rd, w, gp))
    fast = np.asarray(fast)
    monkeypatch.setenv("JQMC_ECP_OK_FORCE_REBUILD", "1")
    ref, forced_sm, forced_rb = _sm_ratio_routes(lambda: jqmc_rs.grad_eL_ecp_wrt_lambda(gem, jd, cd, ru, rd, w, gp))
    ref = np.asarray(ref)
    assert forced_sm == 0 and forced_rb > 0, (forced_sm, forced_rb)
    assert n_sm + n_rb == forced_rb, (n_sm, n_rb, forced_rb)
    if not singular:
        assert (n_sm, n_rb) == (forced_rb, 0), (n_sm, n_rb)
    assert np.isfinite(fast).all() and np.isfinite(ref).all()
    rel = RTOL_RS_ECP_PARAMETER_SM_REBUILD_ILL_CONDITIONED if singular else RTOL_RS_ECP_PARAMETER_SM_REBUILD
    scale = max(np.abs(ref).max(), 1.0)
    assert np.abs(fast - ref).max() <= rel * scale, (np.abs(fast - ref).max(), scale)


# ---- DTN kinetic d(T)/d{alpha,beta,gamma}_indep (u/chi/f channels) ----


def _randomize_dtn(ham, seed=7, scale=0.1):
    """Randomize-then-perturb the DTN parameters (committed jdtn fixtures ship all-zero
    alpha/beta/gamma_indep, which makes a DTN kinetic parameter-gradient test vacuous):
    perturb via ``.replace()``."""
    d = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    rng = np.random.default_rng(seed)
    kw = {}
    for k in ("alpha_indep", "beta_indep", "gamma_indep"):
        v = getattr(d, k, None)
        if v is not None:
            kw[k] = rng.normal(scale=scale, size=np.asarray(v).shape)
    return ham.wavefunction_data.jastrow_data.replace(jastrow_dtn_data=d.replace(**kw))


def _assert_dtn_liveness(d):
    """Guard against the fixture's use_u/use_chi/use_f flags going all-False,
    which would make every ``if d.use_*:`` branch below a silent no-op and the
    whole test vacuously pass with zero blocks actually compared."""
    assert d.use_u or d.use_chi or d.use_f, (
        "DTN fixture has use_u/use_chi/use_f all False -- no block was compared, this test would otherwise pass vacuously"
    )


def _j3_dtn_input(pattern, *, seed=1, dtn_scale=0.1):
    """Canonical J1+J2+J3+DTN interaction on molecular and Gamma ECP inputs.

    Use the reported u8/chi8/f4 orders so every DTN channel is represented.
    Conditioning depends only on the XLA geminal, never parity outcomes.
    """
    from jqmc.jastrow_factor import Jastrow_DTN_data

    ham = _ham(pattern)
    structure, coulomb = ham.structure_data, ham.coulomb_potential_data
    numbers = np.asarray(structure.atomic_numbers)
    effective = np.asarray(coulomb._effective_charges)
    species = np.unique(numbers)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=tuple(float(np.mean(effective[numbers == z])) for z in species),
        pseudized_per_species=tuple(bool(np.all(coulomb.pseudized_per_atom[numbers == z])) for z in species),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=8,
        n_order_chi=8,
        n_order_eN=4,
        n_order_ee=4,
        random_init=True,
        seed=13,
    )
    wf = ham.wavefunction_data
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_dtn_data=dtn)))
    jd = _randomize_dtn(ham, scale=dtn_scale).replace(jastrow_three_body_data=_rand_j3(ham).jastrow_three_body_data)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    ham.sanity_check()
    assert jd.jastrow_three_body_data is not None
    assert np.any(np.asarray(jd.jastrow_three_body_data.j_matrix) != 0)
    assert all(np.any(np.asarray(getattr(jd.jastrow_dtn_data, k)) != 0) for k in ("alpha_indep", "beta_indep", "gamma_indep"))
    ru, rd = draw_well_conditioned_cfg(ham, seed=seed, scale=1.0, cond_max=100.0)
    return ham, jd, ru, rd


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("mol_cart_sd_jdtn", marks=pytest.mark.light),
        pytest.param("mol_ecp_sphe_jdtn", marks=pytest.mark.light),
        pytest.param("mol_ecp_sphe_j3dtn", marks=pytest.mark.heavy),
        pytest.param("pbc_gamma_sphepgto_j3dtn", marks=pytest.mark.heavy),
    ],
)
def test_grad_el_dtn_kinetic(pattern):
    """Grad el DTN kinetic."""
    if pattern.endswith("j3dtn"):
        ham, jd, ru, rd = _j3_dtn_input(pattern)
    else:
        ham = _ham(pattern)
        jd = _randomize_dtn(ham)
        ru, rd = _config(ham)
    d = jd.jastrow_dtn_data
    gem = ham.wavefunction_data.geminal_data
    a, b, g = jqmc_rs.grad_eL_dtn_kinetic_params(gem, jd, ru, rd)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")

    def ref(field, val):
        def fwd(x):
            jd2 = jd.replace(jastrow_dtn_data=d.replace(**{field: x}))
            return CK(Wavefunction_data(jastrow_data=jd2, geminal_data=gem), ru, rd)

        return jax.grad(fwd)(np.asarray(val))

    _assert_dtn_liveness(d)
    if d.use_u:
        comparison_actual_1, comparison_reference_1 = np.asarray(a), np.asarray(ref("alpha_indep", d.alpha_indep))
        if comparison_actual_1.size or comparison_reference_1.size:
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
    if d.use_chi:
        comparison_actual_2, comparison_reference_2 = np.asarray(b), np.asarray(ref("beta_indep", d.beta_indep))
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0
    if d.use_f:
        comparison_actual_3, comparison_reference_3 = np.asarray(g), np.asarray(ref("gamma_indep", d.gamma_indep))
        if comparison_actual_3.size or comparison_reference_3.size:
            np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_3.size == comparison_reference_3.size == 0


@pytest.mark.light
def test_grad_ecp_radial():
    """Basis (geminal radial exp/coeff) e_L on molecular ECP jnone: the Rust
    d(T_det)/d(radial) (jnone kinetic) + d(V_ecp_nl)/d(radial) (mesh-O_k det) vs the
    XLA-analytic Basis e_L c-axis leaf (cart-GTO only)."""
    import jax.numpy as jnp

    from jqmc import derivatives

    ham = _build_ham_for_cell("H2_ecp_ccpvdz_cart.h5", False, None, "SD")  # cart-GTO ECP jnone
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    rz = jqmc_rs.grad_eL_wrt_radial(gem, ru, rd)
    erz = jqmc_rs.grad_eL_ecp_wrt_radial(gem, jd, cd, ru, rd, w, gp)
    xla = derivatives.compute_grad_e_L(
        ham, ru, rd, jnp.eye(3), {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True}, analytic=True
    )["c"]
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    for side in ("orb_data_up_spin", "orb_data_dn_spin"):
        aZ = np.asarray(rz[side][0]) + np.asarray(erz[side][0])
        ac = np.asarray(rz[side][1]) + np.asarray(erz[side][1])
        orb = getattr(xla.geminal_data, side)
        rad = (orb.aos_data if type(orb).__name__ == "MOs_data" else orb).radial
        comparison_actual_1, comparison_reference_1 = np.asarray(aZ), np.asarray(rad.exponents)
        if comparison_actual_1.size or comparison_reference_1.size:
            comparison_label = f"{side}"  # noqa: F841 - captured by the GDD iteration receipt
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
        comparison_actual_2, comparison_reference_2 = np.asarray(ac), np.asarray(rad.coefficients)
        if comparison_actual_2.size or comparison_reference_2.size:
            comparison_label = f"{side}"  # noqa: F841 - captured by the GDD iteration receipt
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
def test_grad_ecp_radial_periodic():
    """Basis (geminal radial exp/coeff) e_L on the PERIODIC real-pGTO ECP jnone
    leaf: d(T_det)/d(radial) (home-cell-wrapped, image-folded jnone kinetic) +
    d(V_ecp_nl)/d(radial) (min-image mesh-O_k det) vs the XLA-analytic Basis e_L
    c-axis leaf. Medium zone (pGTO far-image cutoff / min-image approximation)."""
    import jax.numpy as jnp

    from jqmc import derivatives

    ham = rs_ham("pbc_gamma_sphepgto_jnone")  # periodic real-Gamma sphe-pGTO ECP jnone
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    rz = jqmc_rs.grad_eL_wrt_radial(gem, ru, rd)
    erz = jqmc_rs.grad_eL_ecp_wrt_radial(gem, jd, cd, ru, rd, w, gp)
    xla = derivatives.compute_grad_e_L(
        ham, ru, rd, jnp.eye(3), {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True}, analytic=True
    )["c"]
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    for side in ("orb_data_up_spin", "orb_data_dn_spin"):
        aZ = np.asarray(rz[side][0]) + np.asarray(erz[side][0])
        ac = np.asarray(rz[side][1]) + np.asarray(erz[side][1])
        orb = getattr(xla.geminal_data, side)
        rad = (orb.aos_data if type(orb).__name__ == "MOs_data" else orb).radial
        comparison_actual_1, comparison_reference_1 = np.asarray(aZ), np.asarray(rad.exponents)
        if comparison_actual_1.size or comparison_reference_1.size:
            comparison_label = f"{side}"
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
        comparison_actual_2, comparison_reference_2 = np.asarray(ac), np.asarray(rad.coefficients)
        if comparison_actual_2.size or comparison_reference_2.size:
            comparison_label = f"{side}"
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0


# ---- ECP non-local d(V_ecp)/dc for the Jastrow / DTN blocks (mesh-O_k identity;
#      completes d(e_L)/dc = d(T)/dc + d(V_ecp_nl)/dc for ECP systems) ----


@pytest.mark.light
def test_grad_ecp_jastrow_12():
    """Grad ECP jastrow 12."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

    ham = _ham("mol_ecp_sphe_j12")
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    w, gp = _one_move_mesh()
    RT = np.eye(3)
    ru, rd = _config(ham)
    atol, rtol = _backend_consistency_tolerance(_ECP_ZONES, level="medium")

    def sumV(d):
        return compute_ecp_non_local_parts_nearest_neighbors(
            cd, Wavefunction_data(jastrow_data=d, geminal_data=gem), ru, rd, RT
        )[3]

    a1 = jqmc_rs.grad_eL_ecp_wrt_jastrow_1b_param(gem, jd, cd, ru, rd, w, gp)
    r1 = jax.grad(sumV)(jd).jastrow_one_body_data.jastrow_1b_param
    comparison_actual_1, comparison_reference_1 = np.asarray(a1), np.asarray(r1)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    a2 = jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param(gem, jd, cd, ru, rd, w, gp)
    r2 = jax.grad(sumV)(jd).jastrow_two_body_data.jastrow_2b_param
    comparison_actual_2, comparison_reference_2 = np.asarray(a2), np.asarray(r2)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
def test_grad_ecp_j3():
    """Grad ECP j3."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

    ham = _ham("mol_ecp_sphe_j123")
    jd = _rand_j3(ham)
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    w, gp = _one_move_mesh()
    RT = np.eye(3)
    ru, rd = _config(ham)
    ana = jqmc_rs.grad_eL_ecp_wrt_jastrow_3b_jmatrix(gem, jd, cd, ru, rd, w, gp)
    ref = jax.grad(
        lambda d: compute_ecp_non_local_parts_nearest_neighbors(
            cd, Wavefunction_data(jastrow_data=d, geminal_data=gem), ru, rd, RT
        )[3]
    )(jd).jastrow_three_body_data.j_matrix
    atol, rtol = _backend_consistency_tolerance(_ECP_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("mol_ecp_sphe_jdtn", marks=pytest.mark.light),
        pytest.param("mol_ecp_sphe_j3dtn", marks=pytest.mark.heavy),
        pytest.param("pbc_gamma_sphepgto_j3dtn", marks=pytest.mark.heavy),
    ],
)
def test_grad_ecp_dtn(pattern):
    """Grad ECP DTN."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

    if pattern.endswith("j3dtn"):
        ham, jd, ru, rd = _j3_dtn_input(pattern)
    else:
        ham = _ham(pattern)
        jd = _randomize_dtn(ham)
        ru, rd = draw_well_conditioned_cfg(ham, seed=1, scale=1.0, cond_max=100.0)
    d = jd.jastrow_dtn_data
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    w, gp = _one_move_mesh()
    RT = np.eye(3)
    # Keep cancellation in the mixed XLA autodiff oracle below the established
    # medium parity floor; selection is based only on XLA cond(G).
    ru, rd = draw_well_conditioned_cfg(ham, seed=1, scale=1.0, cond_max=100.0)
    a, b, g = jqmc_rs.grad_eL_ecp_dtn_params(gem, jd, cd, ru, rd, w, gp)
    atol, rtol = _backend_consistency_tolerance(_ECP_ZONES, level="medium")

    def ref(field, val):
        def fwd(x):
            jd2 = jd.replace(jastrow_dtn_data=d.replace(**{field: x}))
            return compute_ecp_non_local_parts_nearest_neighbors(
                cd, Wavefunction_data(jastrow_data=jd2, geminal_data=gem), ru, rd, RT
            )[3]

        return jax.grad(fwd)(np.asarray(val))

    _assert_dtn_liveness(d)
    if d.use_u:
        comparison_actual_1, comparison_reference_1 = np.asarray(a), np.asarray(ref("alpha_indep", d.alpha_indep))
        if comparison_actual_1.size or comparison_reference_1.size:
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
    if d.use_chi:
        comparison_actual_2, comparison_reference_2 = np.asarray(b), np.asarray(ref("beta_indep", d.beta_indep))
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0
    if d.use_f:
        comparison_actual_3, comparison_reference_3 = np.asarray(g), np.asarray(ref("gamma_indep", d.gamma_indep))
        if comparison_actual_3.size or comparison_reference_3.size:
            np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_3.size == comparison_reference_3.size == 0


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_grad_ecp_dtn_uses_nearest_projectors():
    """Compare only the nonlocal ECP term, with two live projector centres."""
    from jqmc.derivatives.assemble import _dV_ecp_dc

    ham, jd, ru, rd = _j3_dtn_input("mol_cl2_ecp_cart_j3dtn", seed=2, dtn_scale=0.01)
    assert len(ham.structure_data.positions) == 2
    assert_fixture_has_live_high_l_channel(ham)
    gem = ham.wavefunction_data.geminal_data
    weights, points = _one_move_mesh()
    alpha, beta, gamma = jqmc_rs.grad_eL_ecp_dtn_params(gem, jd, ham.coulomb_potential_data, ru, rd, weights, points)
    gradient = jax.jit(_dV_ecp_dc)(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.eye(3))
    reference = gradient.jastrow_data.jastrow_dtn_data
    assert all(block is not None and np.asarray(block).size for block in (alpha, beta, gamma))
    atol, rtol = _backend_consistency_tolerance(_ECP_ZONES, level="medium")
    np.testing.assert_allclose(alpha, reference.alpha_indep, atol=atol, rtol=rtol, err_msg="DTN u: nearest ECP projectors")
    np.testing.assert_allclose(beta, reference.beta_indep, atol=atol, rtol=rtol, err_msg="DTN chi: nearest ECP projectors")
    np.testing.assert_allclose(gamma, reference.gamma_indep, atol=atol, rtol=rtol, err_msg="DTN f: nearest ECP projectors")


# ---- PERIODIC (pbc_gamma) + ECP COMBINED d(e_L)/dc: the production lithium/silicon
#      pbc-ECP VMC-optimization case. Full d(e_L)/dc = periodic d(T)/dc + min-image
#      mesh d(V_ecp_nl)/dc, vs the XLA-analytic c-axis leaf on the lithium pbc-ECP ham.
#      Tolerance = medium (periodic pGTO far-image cutoff ~1e-6). ----


def _with_jd(ham, jd):
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


def _grad_eL_c(ham, ru, rd, flag):
    import jax.numpy as jnp

    from jqmc import derivatives

    return derivatives.compute_grad_e_L(ham, ru, rd, jnp.eye(3), flag, analytic=True)["c"]


@pytest.mark.light
def test_grad_ecp_periodic_det_jastrow_12():
    """Grad ECP periodic det jastrow 12."""
    ham = _ham("pbc_gamma_sphepgto_j12")  # lithium pbc-ECP SD j12
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")

    # Det (mo_coefficients): periodic kinetic + min-image ECP mesh.
    ku = jqmc_rs.grad_eL_wrt_mo_coeff(gem, jd, ru, rd)
    eu = jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gem, jd, cd, ru, rd, w, gp)
    au = np.asarray(ku[0]) + np.asarray(eu[0])
    ad = np.asarray(ku[1]) + np.asarray(eu[1])
    xdet = _grad_eL_c(ham, ru, rd, {"opt_geminal": True})
    comparison_actual_1, comparison_reference_1 = np.asarray(au), np.asarray(xdet.geminal_data.orb_data_up_spin.mo_coefficients)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    comparison_actual_2, comparison_reference_2 = np.asarray(ad), np.asarray(xdet.geminal_data.orb_data_dn_spin.mo_coefficients)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0

    # J1 / J2.
    a1 = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_1b_param(gem, jd, ru, rd)) + np.asarray(
        jqmc_rs.grad_eL_ecp_wrt_jastrow_1b_param(gem, jd, cd, ru, rd, w, gp)
    )
    r1 = _grad_eL_c(ham, ru, rd, {"opt_J1_param": True}).jastrow_data.jastrow_one_body_data.jastrow_1b_param
    comparison_actual_3, comparison_reference_3 = np.asarray(a1), np.asarray(r1)
    if comparison_actual_3.size or comparison_reference_3.size:
        np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_3.size == comparison_reference_3.size == 0
    a2 = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_2b_param(gem, jd, ru, rd)) + np.asarray(
        jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param(gem, jd, cd, ru, rd, w, gp)
    )
    r2 = _grad_eL_c(ham, ru, rd, {"opt_J2_param": True}).jastrow_data.jastrow_two_body_data.jastrow_2b_param
    comparison_actual_4, comparison_reference_4 = np.asarray(a2), np.asarray(r2)
    if comparison_actual_4.size or comparison_reference_4.size:
        np.testing.assert_allclose(comparison_actual_4, comparison_reference_4, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_4.size == comparison_reference_4.size == 0


@pytest.mark.light
def test_grad_ecp_periodic_j3():
    # Oracle = ``jax.grad(compute_local_energy)``: the derivative of the production
    # forward, hence unambiguously the true dE_L/dc. (An earlier revision of this comment
    # claimed jqmc's XLA-analytic route was WRONG here -- that was FALSE. It arose from
    # injecting a NON-SYMMETRIC j_matrix, which ``Jastrow_three_body_data.sanity_check``
    # rejects and which several jqmc kernels are designed to exclude
    # (``j3_mat.T @ x == j3_mat @ x``). ``_rand_j3`` symmetrizes, matching production;
    # with a valid J the analytic route agrees with ``_auto`` at ~1e-15.)
    """Grad ECP periodic j3."""
    import jax.numpy as jnp

    from jqmc.hamiltonians import compute_local_energy

    ham0 = _ham("pbc_gamma_sphepgto_j123")
    jd = _rand_j3(ham0)  # committed j123 ships zeros
    ham = _with_jd(ham0, jd)
    gem = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    ana = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_3b_jmatrix(gem, jd, ru, rd)) + np.asarray(
        jqmc_rs.grad_eL_ecp_wrt_jastrow_3b_jmatrix(gem, jd, cd, ru, rd, w, gp)
    )
    ref = jax.grad(lambda d: compute_local_energy(_with_jd(ham, d), ru, rd, jnp.eye(3)))(jd).jastrow_three_body_data.j_matrix
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


@pytest.mark.light
def test_grad_ecp_periodic_dtn():
    """Grad ECP periodic DTN."""
    ham0 = _ham("pbc_gamma_sphepgto_jdtn")
    jd = _randomize_dtn(ham0)  # committed jdtn ships zeros
    ham = _with_jd(ham0, jd)
    d = jd.jastrow_dtn_data
    gem = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    ka, kb, kg = (None if x is None else np.asarray(x) for x in jqmc_rs.grad_eL_dtn_kinetic_params(gem, jd, ru, rd))
    ea, eb, eg = jqmc_rs.grad_eL_ecp_dtn_params(gem, jd, cd, ru, rd, w, gp)
    a = ka if ea is None else ka + np.asarray(ea)
    b = kb if eb is None else kb + np.asarray(eb)
    g = kg if eg is None else kg + np.asarray(eg)
    xd = _grad_eL_c(
        ham, ru, rd, {"opt_JDTN_u_param": True, "opt_JDTN_chi_param": True, "opt_JDTN_f_param": True}
    ).jastrow_data.jastrow_dtn_data
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    _assert_dtn_liveness(d)
    if d.use_u:
        comparison_actual_1, comparison_reference_1 = np.asarray(a), np.asarray(xd.alpha_indep)
        if comparison_actual_1.size or comparison_reference_1.size:
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
    if d.use_chi:
        comparison_actual_2, comparison_reference_2 = np.asarray(b), np.asarray(xd.beta_indep)
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0
    if d.use_f:
        comparison_actual_3, comparison_reference_3 = np.asarray(g), np.asarray(xd.gamma_indep)
        if comparison_actual_3.size or comparison_reference_3.size:
            np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_3.size == comparison_reference_3.size == 0


# ---- COMPLEX-TWIST (k != Gamma) Jastrow-param de_L/dc = Re(dT/dc).
#      At a twist the orbitals are complex but the Jastrow is REAL, so dE_L/dc = Re(dT/dc)
#      via the complex determinant drift; ECP adds the complex-mesh dV_ecp_nl/dc. Oracle =
#      jax.grad(Re(compute_local_energy)) (the unambiguous e_L derivative -- the jqmc
#      analytic route has a known J3+ECP bug). Tolerance = medium (complex pGTO cutoff). ----


def _twist_ref(ham, ru, rd):
    """jax.grad(Re(compute_local_energy)) wrt the jastrow_data pytree, at a twist."""
    import jax.numpy as jnp

    from jqmc.hamiltonians import compute_local_energy

    jd = ham.wavefunction_data.jastrow_data
    return jax.grad(lambda d: jnp.real(compute_local_energy(_with_jd(ham, d), ru, rd, jnp.eye(3))))(jd)


@pytest.mark.heavy
def test_grad_el_twist_ae_j123():
    """AE (non-ECP) twist: J1/J2/J3 dE_L/dc = Re(dT/dc). No V dependence (V is real,
    jastrow-independent), so the whole gradient is the complex-kinetic reduction."""
    ham0 = _build_ham_for_cell("H2_pbc_ae_k025_sphe.h5", True, {"j3": "pgto"}, "SD")
    jd = _rand_j3(ham0)  # committed j123 ships a zero j_matrix
    ham = _with_jd(ham0, jd)
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    ref = _twist_ref(ham, ru, rd)
    a1 = jqmc_rs.grad_eL_wrt_jastrow_1b_param_complex(gem, jd, ru, rd)
    comparison_actual_1, comparison_reference_1 = np.asarray(a1), np.asarray(ref.jastrow_one_body_data.jastrow_1b_param)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    a2 = jqmc_rs.grad_eL_wrt_jastrow_2b_param_complex(gem, jd, ru, rd)
    comparison_actual_2, comparison_reference_2 = np.asarray(a2), np.asarray(ref.jastrow_two_body_data.jastrow_2b_param)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0
    a3 = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_3b_jmatrix_complex(gem, jd, ru, rd))
    r3 = np.asarray(ref.jastrow_three_body_data.j_matrix)
    comparison_actual_3, comparison_reference_3 = np.asarray(a3), np.asarray(r3)
    if comparison_actual_3.size or comparison_reference_3.size:
        np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_3.size == comparison_reference_3.size == 0


@pytest.mark.heavy
def test_grad_el_twist_ecp_j12():
    """ECP twist: J1/J2 dE_L/dc = Re(dT/dc) + dV_ecp_nl/dc (complex mesh, Re-projected)."""
    ham = _ham("twist_sphepgto_j12")  # lithium_k025 pbc-ECP SD j12
    gem = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    ref = _twist_ref(ham, ru, rd)
    a1 = jqmc_rs.grad_eL_wrt_jastrow_1b_param_complex(gem, jd, ru, rd) + jqmc_rs.grad_eL_ecp_wrt_jastrow_1b_param_complex(
        gem, jd, cd, ru, rd, w, gp
    )
    comparison_actual_1, comparison_reference_1 = np.asarray(a1), np.asarray(ref.jastrow_one_body_data.jastrow_1b_param)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    a2 = jqmc_rs.grad_eL_wrt_jastrow_2b_param_complex(gem, jd, ru, rd) + jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param_complex(
        gem, jd, cd, ru, rd, w, gp
    )
    comparison_actual_2, comparison_reference_2 = np.asarray(a2), np.asarray(ref.jastrow_two_body_data.jastrow_2b_param)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.heavy
def test_grad_el_twist_ecp_dtn():
    """ECP twist: DTN u/chi/f dE_L/dc = Re(dT/dc) + complex-mesh dV_ecp_nl/dc."""
    ham0 = _ham("twist_sphepgto_jdtn")  # lithium_k025 pbc-ECP SD jdtn
    jd = _randomize_dtn(ham0)  # committed jdtn ships zeros
    ham = _with_jd(ham0, jd)
    d = jd.jastrow_dtn_data
    gem = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    ref = _twist_ref(ham, ru, rd).jastrow_dtn_data
    ka, kb, kg = (None if x is None else np.asarray(x) for x in jqmc_rs.grad_eL_dtn_kinetic_params_complex(gem, jd, ru, rd))
    ea, eb, eg = jqmc_rs.grad_eL_ecp_dtn_params_complex(gem, jd, cd, ru, rd, w, gp)
    a = ka if ea is None else ka + np.asarray(ea)
    b = kb if eb is None else kb + np.asarray(eb)
    g = kg if eg is None else kg + np.asarray(eg)
    _assert_dtn_liveness(d)
    if d.use_u:
        comparison_actual_1, comparison_reference_1 = np.asarray(a), np.asarray(ref.alpha_indep)
        if comparison_actual_1.size or comparison_reference_1.size:
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
    if d.use_chi:
        comparison_actual_2, comparison_reference_2 = np.asarray(b), np.asarray(ref.beta_indep)
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0
    if d.use_f:
        comparison_actual_3, comparison_reference_3 = np.asarray(g), np.asarray(ref.gamma_indep)
        if comparison_actual_3.size or comparison_reference_3.size:
            np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_3.size == comparison_reference_3.size == 0


@pytest.mark.heavy
@pytest.mark.parametrize(
    "fixture_path, has_ecp",
    [
        pytest.param("tests/trexio_example_files/lithium_k025.h5", True, id="li-ecp"),
        pytest.param("tests/trexio_example_files/H4_pbc_ae_k025_cart.h5", False, id="h4-ae"),
    ],
)
def test_grad_el_twist_dtn_with_j3(fixture_path, has_ecp):
    """DTN energy derivatives include the active real J3 drift at a complex twist."""
    from jqmc.atomic_orbital import is_complex_aos

    ham = _build_ham_for_cell(Path(fixture_path).name, True, {"j3": "pgto", "dtn": True})
    jd = _randomize_dtn(ham).replace(jastrow_three_body_data=_rand_j3(ham).jastrow_three_body_data)
    ham = _with_jd(ham, jd)
    gem = ham.wavefunction_data.geminal_data
    orbitals = gem.orb_data_up_spin.aos_data
    assert is_complex_aos(orbitals) and orbitals.k_point is not None and any(orbitals.k_point)
    assert np.any(np.asarray(jd.jastrow_three_body_data.j_matrix) != 0)
    assert jd.jastrow_dtn_data.use_u and jd.jastrow_dtn_data.use_chi and jd.jastrow_dtn_data.use_f

    cd = ham.coulomb_potential_data
    assert bool(np.any(cd.pseudized_per_atom)) == has_ecp
    if has_ecp:
        ru, rd = _config(ham)
    else:
        # Put one up/down pair near each H2 fragment, inside its DTN f cutoff.
        assert gem.num_electron_up == gem.num_electron_dn == 2
        centres = np.asarray(ham.structure_data.positions)[[0, 2]]
        ru = centres + np.array([[0.5, -0.4, 0.3], [-0.7, 0.5, 0.6]])
        rd = centres + np.array([[-0.3, 0.6, 0.8], [0.6, -0.7, 0.4]])
        assert np.linalg.cond(np.asarray(compute_geminal_all_elements(gem, ru, rd))) < 100

    native = jqmc_rs.grad_eL_dtn_kinetic_params_complex(gem, jd, ru, rd, True)
    if has_ecp:
        w, gp = _one_move_mesh()
        ecp = jqmc_rs.grad_eL_ecp_dtn_params_complex(gem, jd, cd, ru, rd, w, gp, True)
        native = tuple(kinetic + mesh for kinetic, mesh in zip(native, ecp, strict=True))

    ref = _twist_ref(ham, ru, rd).jastrow_dtn_data
    atol, rtol = _backend_consistency_tolerance(_KINETIC_ZONES, level="medium")
    comparison_actual_1, comparison_reference_1 = np.real(native[2]), np.asarray(ref.gamma_indep)
    if not has_ecp:
        # The Li configuration has no active f pair. The H4 configuration must witness it.
        assert np.any(np.abs(comparison_reference_1) > atol), "the H4 DTN f derivative must be nonzero"
    np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    comparison_actual_2, comparison_reference_2 = np.real(native[0]), np.asarray(ref.alpha_indep)
    np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    comparison_actual_3, comparison_reference_3 = np.real(native[1]), np.asarray(ref.beta_indep)
    np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)


# ---- Fast ECP parameter-gradient mesh + exact one-electron O_k increments ----
# Every ECP non-local mesh candidate moves ONE electron. The twist (k != Gamma) kernels
# ride the rank-1 determinant / incremental-Jastrow candidate mesh the native complex
# e_L uses (``MeshFastStateC``) instead of rebuilding the full complex geminal LU and the
# full Jastrow value per candidate, and the J1 / J2 / J3 ``j_matrix`` mesh reduces (real
# and twist) apply the exact change of O_k under that one-electron move instead of
# re-evaluating the full O_k per candidate. ``JQMC_ECP_PARAM_FORCE_REFERENCE`` restores
# the pre-change reference (full-recompute mesh + per-candidate full O_k) on identical
# inputs, so each case below is an A/B parity check against it, plus a complexity guard
# read from ``jqmc_rs.route_predicate_counters()``.

# route_predicate_counters indices (append-only; see gfmc::route_predicate_counters).
_PARAM_FAST_MESHES = 12
_PARAM_REF_MESH_CANDIDATES = 13
_PARAM_OK_INCREMENTAL = 14
_PARAM_OK_FULL_EVALS = 15


def _twist_ecp_call(kernel):
    def call(gem, jd, cd, ru, rd, w, gp):
        return getattr(jqmc_rs, kernel)(gem, jd, cd, ru, rd, w, gp, True)

    return call


def _ecp_call(kernel):
    def call(gem, jd, cd, ru, rd, w, gp):
        return getattr(jqmc_rs, kernel)(gem, jd, cd, ru, rd, w, gp)

    return call


# case -> (fixture pattern, kernel call, twist mesh?, J1/J2/J3 j_matrix O_k reduce?)
_ECP_INCREMENTAL_CASES = {
    "twist_j1": ("twist_sphepgto_j123", _twist_ecp_call("grad_eL_ecp_wrt_jastrow_1b_param_complex"), True, True),
    "twist_j2": ("twist_sphepgto_j123", _twist_ecp_call("grad_eL_ecp_wrt_jastrow_2b_param_complex"), True, True),
    "twist_j3": ("twist_sphepgto_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex"), True, True),
    "twist_j3_radial": ("twist_sphepgto_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_3b_radial_complex"), True, False),
    "twist_geminal_radial": ("twist_sphepgto_j123", _twist_ecp_call("grad_eL_ecp_wrt_radial_complex"), True, False),
    "twist_dtn": ("twist_sphepgto_jdtn", _twist_ecp_call("grad_eL_ecp_dtn_params_complex"), True, False),
    "gamma_j1": ("pbc_gamma_sphepgto_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_1b_param"), False, True),
    "gamma_j2": ("pbc_gamma_sphepgto_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_2b_param"), False, True),
    "gamma_j3": ("pbc_gamma_sphepgto_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_3b_jmatrix"), False, True),
    "mol_j1": ("mol_ecp_sphe_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_1b_param"), False, True),
    "mol_j2": ("mol_ecp_sphe_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_2b_param"), False, True),
    "mol_j3": ("mol_ecp_sphe_j123", _ecp_call("grad_eL_ecp_wrt_jastrow_3b_jmatrix"), False, True),
}


_ECP_INCREMENTAL_LATTICE_CASES = {
    "twist_j2_lattice_shifted": ("twist_sphepgto_j12", _twist_ecp_call("grad_eL_ecp_wrt_jastrow_2b_param_complex"), True, True),
    "twist_lambda_lattice_shifted": ("twist_agp_jnone", _ecp_call("grad_eL_ecp_wrt_lambda_complex"), True, False),
}


def _flatten_payload(x):
    """Flatten a kernel payload (scalar / array / nested tuple / dict, possibly with None
    DTN channels) into one complex vector, in a deterministic order."""
    if x is None:
        return np.zeros(0, dtype=np.complex128)
    if isinstance(x, dict):
        return np.concatenate([_flatten_payload(x[k]) for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        parts = [_flatten_payload(v) for v in x]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.complex128)
    return np.asarray(x, dtype=np.complex128).ravel()


def _incremental_routes(call):
    """Run ``call`` in a fresh counter window; return (flat payload, route counters)."""
    jqmc_rs.gfmc_route_counters_reset()
    out = _flatten_payload(call())
    c = list(jqmc_rs.route_predicate_counters())
    return out, {
        "fast_meshes": c[_PARAM_FAST_MESHES],
        "ref_mesh_candidates": c[_PARAM_REF_MESH_CANDIDATES],
        "ok_incremental": c[_PARAM_OK_INCREMENTAL],
        "ok_full_evals": c[_PARAM_OK_FULL_EVALS],
    }


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_complex_twist_ecp_j2_gradient_uses_production_nearest_nuclei(monkeypatch):
    """Both native mesh paths differentiate the default nonlocal ECP operator.

    Use the committed non-Gamma lithium wavefunction with a wider nonlocal
    projector reach so the second centre changes the J2 derivative by more
    than the backend-consistency budget in either precision.
    """
    from jqmc.coulomb_potential import (
        compute_ecp_non_local_parts_nearest_neighbors_holomorphic,
        icosahedron_sym_mesh_Nv12,
    )

    ham = _ham("twist_sphepgto_j123")
    wf, coulomb = ham.wavefunction_data, ham.coulomb_potential_data
    coulomb = coulomb.replace(
        exponents=tuple(
            0.1 * float(exponent) if ell >= 0 else float(exponent)
            for exponent, ell in zip(coulomb.exponents, coulomb.ang_moms, strict=True)
        )
    )
    coulomb.sanity_check()
    centres = np.asarray(ham.structure_data.positions)
    assert centres.shape == (2, 3)
    separation = centres[1] - centres[0]
    ru = (centres[0] + 0.45 * separation + np.array([0.09, -0.13, 0.07]))[None, :]
    rd = (centres[0] + 0.36 * separation + np.array([-0.12, 0.07, 0.03]))[None, :]
    mesh = icosahedron_sym_mesh_Nv12
    weights, grid = np.asarray(mesh.weights), np.asarray(mesh.grid_points)
    jastrow = wf.jastrow_data
    j2 = jastrow.jastrow_two_body_data

    def nonlocal_energy(j2_param, **kwargs):
        updated = wf.replace(jastrow_data=jastrow.replace(jastrow_two_body_data=j2.replace(jastrow_2b_param=j2_param)))
        return compute_ecp_non_local_parts_nearest_neighbors_holomorphic(coulomb, updated, ru, rd, jnp.eye(3), **kwargs)[3].real

    derivative = jax.grad(nonlocal_energy)
    expected = float(derivative(j2.jastrow_2b_param))
    all_pairs = float(derivative(j2.jastrow_2b_param, NN=len(centres)))
    zones = ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval")
    atol, rtol = _backend_consistency_tolerance(zones, level="medium")
    assert np.isfinite([expected, all_pairs]).all()
    assert abs(expected) > atol
    assert abs(expected - all_pairs) > atol + rtol * abs(expected), "the second centre must affect the derivative"

    def native():
        return jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param_complex(wf.geminal_data, jastrow, coulomb, ru, rd, weights, grid)

    monkeypatch.delenv("JQMC_ECP_PARAM_FORCE_REFERENCE", raising=False)
    fast, fast_routes = _incremental_routes(native)
    assert fast_routes["fast_meshes"] == 1 and fast_routes["ref_mesh_candidates"] == 0
    monkeypatch.setenv("JQMC_ECP_PARAM_FORCE_REFERENCE", "1")
    rebuilt, rebuilt_routes = _incremental_routes(native)
    assert rebuilt_routes["fast_meshes"] == 0 and rebuilt_routes["ref_mesh_candidates"] > 0
    actual = np.concatenate([fast.real, rebuilt.real])
    reference = np.full(2, expected)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


def _incremental_inputs(case, *, home=False):
    pattern, call, twist, jastrow_ok = (_ECP_INCREMENTAL_CASES | _ECP_INCREMENTAL_LATTICE_CASES)[case]
    ham = _ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    if case == "twist_dtn":
        # Committed jdtn fixtures ship all-zero DTN parameters; perturb them so the
        # DTN ratio entering every mesh weight is live.
        jd = _randomize_dtn(ham)
    gem = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    w, gp = _one_move_mesh()
    ru, rd = _config(ham)
    if case.endswith("_lattice_shifted"):
        # Independent lattice invariance: a Bloch phase acquired by translating
        # one electron cancels from E_L and its wavefunction-parameter derivatives.
        # Exhaustive image search proves that raw Cartesian ranking would select
        # a different ECP nucleus; no production minimum-image helper is used.
        assert (int(gem.num_electron_up), int(gem.num_electron_dn)) == (1, 1)
        nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
        cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
        rng = np.random.default_rng(5)
        home_up = (nuclei[0] + rng.normal(scale=0.3, size=3))[None]
        home_dn = (nuclei[1] + rng.normal(scale=0.3, size=3))[None]
        # Translate one electron: the other stays in the home cell. This
        # well-conditioned witness separates the nucleus rankings without
        # amplifying finite-image AO truncation in the AGP parameter payload.
        ru = home_up + np.array([1, 1, 0]) @ cell
        rd = home_dn
        translated = np.vstack([ru, rd])
        near = np.argmin(min_image_distances(nuclei, cell, translated), axis=1)
        raw = np.argmin(np.linalg.norm(nuclei[None] - translated[:, None], axis=-1), axis=1)
        np.testing.assert_array_equal(near, [0, 1])
        assert np.any(raw != near), "geometry must distinguish raw and periodic nearest nuclei"
        home_raw = np.argmin(np.linalg.norm(nuclei[None] - np.vstack([home_up, home_dn])[:, None], axis=-1), axis=1)
        np.testing.assert_array_equal(home_raw, near)
        if home:
            ru, rd = home_up, home_dn
    return (lambda: call(gem, jd, cd, ru, rd, w, gp)), twist, jastrow_ok


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_ECP_INCREMENTAL_CASES))
def test_ecp_param_grad_fast_vs_reference(case, monkeypatch):
    """Parity: the fast twist mesh + exact one-electron O_k increments equal the
    full-recompute reference (``JQMC_ECP_PARAM_FORCE_REFERENCE``) on identical inputs.

    Both routes evaluate the same mesh identity ``Sum_g v_g [O_k(r'_g) - O_k(r)]``; the
    rank-1 determinant ratio, the incremental Jastrow ratio and the one-electron O_k
    increment change only the floating-point association. Full precision: the gap is held
    to the SM-vs-rebuild family ``RTOL_RS_ECP_PARAMETER_SM_REBUILD`` of the payload scale.
    Mixed precision: the fast mesh evaluates its candidate ratios in the active fp32
    zones while the reference rebuilds them, so the ordinary strict budget of the ECP
    zones (``get_tolerance_min(_ECP_ZONES)``) governs, judged within the mixed lane.
    """
    call, _twist, _jastrow_ok = _incremental_inputs(case)
    monkeypatch.delenv("JQMC_ECP_PARAM_FORCE_REFERENCE", raising=False)
    fast, _ = _incremental_routes(call)
    monkeypatch.setenv("JQMC_ECP_PARAM_FORCE_REFERENCE", "1")
    ref, _ = _incremental_routes(call)
    assert fast.shape == ref.shape and fast.size > 0, (fast.shape, ref.shape)
    assert np.isfinite(fast).all() and np.isfinite(ref).all()
    assert np.abs(ref).max() > 0.0, "vacuous case: the reference payload is identically zero"
    scale = max(np.abs(ref).max(), 1.0)
    gap = np.abs(fast - ref).max()
    if is_mixed_precision_enabled():
        atol, rtol = get_tolerance_min(_ECP_ZONES, "strict")
        bound = atol + rtol * scale
    else:
        bound = RTOL_RS_ECP_PARAMETER_SM_REBUILD * scale
    assert gap <= bound, (case, gap, scale, bound)


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_ECP_INCREMENTAL_CASES))
def test_ecp_param_grad_route_complexity(case, monkeypatch):
    """Complexity guard: one ECP parameter-gradient call performs O(1) full
    Jastrow / O_k evaluations, not O(N_electron * N_quadrature).

    Fast route: the twist mesh is served by the fast mesh state (one event, zero
    reference-mesh candidates, each of which would cost a full Jastrow value and a full
    complex geminal determinant), and a J1 / J2 / J3 ``j_matrix`` reduce is incremental
    (one event, zero full O_k evaluations). The forced reference proves the counters
    observe the expensive route: one full-recompute candidate per mesh point and
    ``n_candidates + 1`` full O_k evaluations.
    """
    call, twist, jastrow_ok = _incremental_inputs(case)
    monkeypatch.delenv("JQMC_ECP_PARAM_FORCE_REFERENCE", raising=False)
    _, fast = _incremental_routes(call)
    monkeypatch.setenv("JQMC_ECP_PARAM_FORCE_REFERENCE", "1")
    _, ref = _incremental_routes(call)
    assert fast["ref_mesh_candidates"] == 0, fast
    assert fast["ok_full_evals"] == 0, fast
    assert fast["fast_meshes"] == (1 if twist else 0), fast
    assert fast["ok_incremental"] == (1 if jastrow_ok else 0), fast
    if twist:
        assert ref["fast_meshes"] == 0 and ref["ref_mesh_candidates"] > 0, ref
    if jastrow_ok:
        assert ref["ok_incremental"] == 0 and ref["ok_full_evals"] > 1, ref
    if twist and jastrow_ok:
        # Reference reduce = one full O_k per mesh candidate plus the base evaluation.
        assert ref["ok_full_evals"] == ref["ref_mesh_candidates"] + 1, ref


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_ECP_INCREMENTAL_LATTICE_CASES))
def test_ecp_param_grad_reference_mesh_lattice_translation(case, monkeypatch):
    """Pin reference-mesh NN=1 ranking with independent Bloch lattice invariance.

    Preserve the original parameter-family cases and their GDD node contracts.
    These off-cell witnesses add both physical and fast/reference checks under
    the same prescribed precision budgets, with counters proving actual routes.
    """
    call, _, incremental = _incremental_inputs(case)
    monkeypatch.delenv("JQMC_ECP_PARAM_FORCE_REFERENCE", raising=False)
    fast, fast_routes = _incremental_routes(call)
    monkeypatch.setenv("JQMC_ECP_PARAM_FORCE_REFERENCE", "1")
    reference, reference_routes = _incremental_routes(call)
    home_call, _, _ = _incremental_inputs(case, home=True)
    home_reference, home_routes = _incremental_routes(home_call)
    assert fast.shape == reference.shape == home_reference.shape and reference.size > 0
    for payload in (fast, reference, home_reference):
        assert np.isfinite(payload).all()
        assert np.max(np.abs(payload)) > 0.0
    assert fast_routes["fast_meshes"] == 1 and fast_routes["ref_mesh_candidates"] == 0, fast_routes
    assert fast_routes["ok_full_evals"] == 0 and fast_routes["ok_incremental"] == int(incremental), fast_routes
    for routes in (reference_routes, home_routes):
        assert routes["fast_meshes"] == 0 and routes["ref_mesh_candidates"] > 0, routes
    if incremental:
        assert fast_routes["ok_incremental"] == 1 and fast_routes["ok_full_evals"] == 0, fast_routes
        for routes in (reference_routes, home_routes):
            assert routes["ok_full_evals"] == routes["ref_mesh_candidates"] + 1, routes
    # Fast/reference equality and home/translated Bloch invariance are checked
    # separately. Each parameter-family payload is a homogeneous complex block.
    for actual, expected in ((fast, reference), (reference, home_reference)):
        scale = max(np.max(np.abs(expected)), 1.0)
        if is_mixed_precision_enabled():
            atol, rtol = get_tolerance_min(_ECP_ZONES, "strict")
            bound = atol + rtol * scale
        else:
            bound = RTOL_RS_ECP_PARAMETER_SM_REBUILD * scale
        assert np.max(np.abs(actual - expected)) <= bound, (case, actual, expected, bound)


# ---- Complex Jastrow c-axis methods on the once-built native context ----
# The free complex kernels re-parse the complex geminal, the Jastrow, the J3 orbital
# and the ECP tables on every walker and family. A context built by
# ``build_obs_ctx(complex=True, want_param_deriv=True)`` (the driver's context)
# parses them once and serves the same kernels as ``NativeObsCtx.pd_*_complex``; the
# adapter routes through them when the context reports ``has_complex_param_deriv()``.
# Same kernels on the same data: the two routes must agree to reduction-order class
# (``get_tolerance_reduction_order``).


def _free_complex(kernel, *, ecp, source_mode):
    def call(gem, jd, cd, ru, rd, w, gp):
        args = (gem, jd, cd, ru, rd, w, gp) if ecp else (gem, jd, ru, rd)
        return getattr(jqmc_rs, kernel)(*args, *((True,) if source_mode else ()))

    return call


# case -> (NativeObsCtx method, free-kernel reference, takes the ECP mesh?)
_NATIVE_CTX_METHODS = {
    "ok_j3": (
        "pd_ok_j3_complex",
        lambda gem, jd, cd, ru, rd, w, gp: jqmc_rs.grad_ln_Psi_jastrow_3b_jmatrix_complex(jd.jastrow_three_body_data, ru, rd),
        False,
    ),
    "el_j1": (
        "pd_el_j1_complex",
        _free_complex("grad_eL_wrt_jastrow_1b_param_complex", ecp=False, source_mode=True),
        False,
    ),
    "el_j2": (
        "pd_el_j2_complex",
        _free_complex("grad_eL_wrt_jastrow_2b_param_complex", ecp=False, source_mode=True),
        False,
    ),
    "el_j3": (
        "pd_el_j3_complex",
        _free_complex("grad_eL_wrt_jastrow_3b_jmatrix_complex", ecp=False, source_mode=False),
        False,
    ),
    "el_ecp_j1": (
        "pd_el_ecp_j1_complex",
        _free_complex("grad_eL_ecp_wrt_jastrow_1b_param_complex", ecp=True, source_mode=True),
        True,
    ),
    "el_ecp_j2": (
        "pd_el_ecp_j2_complex",
        _free_complex("grad_eL_ecp_wrt_jastrow_2b_param_complex", ecp=True, source_mode=True),
        True,
    ),
    "el_ecp_j3": (
        "pd_el_ecp_j3_complex",
        _free_complex("grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex", ecp=True, source_mode=False),
        True,
    ),
}


@pytest.mark.light
@pytest.mark.parametrize("method", sorted(_NATIVE_CTX_METHODS))
def test_complex_pd_adapter_native_ctx_matches_free_kernels(method):
    """The once-built native complex Jastrow tables equal the free kernels.

    Also witnesses the adapter routing: ``_RsComplexPdCtx`` holds the context for the
    Jastrow families only when it reports ``has_complex_param_deriv()``.
    """
    from jqmc._precision import get_tolerance_reduction_order, precision_policy_snapshot
    from jqmc.jqmc_mcmc import _RsComplexPdCtx

    ham = _ham("twist_sphepgto_j123")
    wf = ham.wavefunction_data
    gem, jd, cd = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    policy = precision_policy_snapshot()
    ctx = jqmc_rs.build_obs_ctx(gem, jd, cd, True, True, False, False, False, True, precision_policy=policy)
    plain = jqmc_rs.build_obs_ctx(gem, jd, cd, True, True, False, precision_policy=policy)
    assert ctx.has_complex_param_deriv() and not plain.has_complex_param_deriv()
    assert _RsComplexPdCtx(gem, jd, cd, ctx)._native_jastrow is ctx
    assert _RsComplexPdCtx(gem, jd, cd, plain)._native_jastrow is None
    name, free, ecp = _NATIVE_CTX_METHODS[method]
    ru, rd = _config(ham)
    w, gp = _one_move_mesh()
    got = _flatten_payload(getattr(ctx, name)(ru, rd, w, gp) if ecp else getattr(ctx, name)(ru, rd))
    ref = _flatten_payload(free(gem, jd, cd, ru, rd, w, gp))
    assert got.shape == ref.shape and got.size > 0
    assert np.isfinite(got).all() and np.isfinite(ref).all()
    assert np.abs(ref).max() > 0.0, "vacuous case: the free-kernel payload is identically zero"
    atol, rtol = get_tolerance_reduction_order(_ECP_ZONES, "strict")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)


# Complex-twist Jastrow walker batches (``_pd_batch_base`` / ``_pd_batch_ecp`` on
# ``_RsComplexPdCtx``): method -> (fixture, adapter method, ECP).
_COMPLEX_PD_BATCH_METHODS = {
    "ok_j3": ("twist_sphepgto_j123", "pd_ok_j3", False),
    "el_j1": ("twist_sphepgto_j123", "pd_el_j1", False),
    "el_j2": ("twist_sphepgto_j123", "pd_el_j2", False),
    "el_j3": ("twist_sphepgto_j123", "pd_el_j3", False),
    "el_ecp_j1": ("twist_sphepgto_j123", "pd_el_ecp_j1", True),
    "el_ecp_j2": ("twist_sphepgto_j123", "pd_el_ecp_j2", True),
    "el_ecp_j3": ("twist_sphepgto_j123", "pd_el_ecp_j3", True),
    "ok_dtn": ("twist_sphepgto_jdtn", "pd_ok_dtn", False),
    "el_dtn": ("twist_sphepgto_jdtn", "pd_el_dtn", False),
    "el_ecp_dtn": ("twist_sphepgto_jdtn", "pd_el_ecp_dtn", True),
}


@pytest.mark.light
@pytest.mark.parametrize("method", sorted(_COMPLEX_PD_BATCH_METHODS))
def test_complex_pd_adapter_walker_batch_matches_per_walker(method):
    """The complex-twist adapter's walker batch equals its scalar method walker by walker.

    With the native context the batch runs on the context's rayon ``*_complex_batch``
    twin (the driver path). Each walker gets its own configuration and its own rotated
    ECP mesh, so a mis-indexed walker or mesh fails.
    """
    from jqmc._precision import get_tolerance_reduction_order, precision_policy_snapshot
    from jqmc.jqmc_mcmc import _pd_batch_base, _pd_batch_ecp, _pd_stack, _RsComplexPdCtx

    fixture, name, ecp = _COMPLEX_PD_BATCH_METHODS[method]
    ham = _ham(fixture)
    wf = ham.wavefunction_data
    gem, jd, cd = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ctx = jqmc_rs.build_obs_ctx(
        gem, jd, cd, True, True, False, False, False, True, precision_policy=precision_policy_snapshot()
    )
    adapter = _RsComplexPdCtx(gem, jd, cd, ctx)
    assert adapter._native_jastrow is ctx and adapter._native_ctx is ctx
    configs = [_config(ham, seed=s) for s in (1, 2, 3)]
    ru_b = np.stack([c[0] for c in configs])
    rd_b = np.stack([c[1] for c in configs])
    w, gp = _one_move_mesh()
    rot = [np.linalg.qr(np.random.default_rng(s).normal(size=(3, 3)))[0] for s in (4, 5, 6)]
    gp_b = np.stack([gp @ (q * np.sign(np.linalg.det(q))) for q in rot])
    if ecp:
        ref = _pd_stack([getattr(adapter, name)(ru_b[k], rd_b[k], w, gp_b[k]) for k in range(3)])
    else:
        ref = _pd_stack([getattr(adapter, name)(ru_b[k], rd_b[k]) for k in range(3)])

    got = _pd_batch_ecp(adapter, name, ru_b, rd_b, w, gp_b) if ecp else _pd_batch_base(adapter, name, ru_b, rd_b)
    got_c, ref_c = _flatten_payload(got), _flatten_payload(ref)
    assert got_c.shape == ref_c.shape and got_c.size > 0
    assert np.isfinite(got_c).all() and np.isfinite(ref_c).all()
    assert np.abs(ref_c).max() > 0.0, "vacuous case: the per-walker payload is identically zero"
    walkers = [_flatten_payload(x) for x in (ref if isinstance(ref, tuple) else (ref,))]
    assert any(not np.array_equal(v.reshape(3, -1)[0], v.reshape(3, -1)[1]) for v in walkers if v.size), "walkers coincide"
    atol, rtol = get_tolerance_reduction_order(_ECP_ZONES, "strict")
    np.testing.assert_allclose(got_c, ref_c, atol=atol, rtol=rtol)

"""Layer-1 equivalence: closed-form parameter gradients O_k = d(ln Psi)/dc (Rust)
vs jax.grad(jqmc forward) -- the production `_auto` oracle. Covers geminal
lambda / mo_coefficients, Jastrow J1/J2/J3 j_matrix, DTN alpha/beta/gamma.
All real / Gamma-point, molecular (CART + SPHE).

Hamiltonians come from the committed-fixture builders
(``tests.test_rs_kernel._rs_fixtures.rs_ham``) and tolerances from the dedicated
XLA--Rust consistency helper.

Two local pattern extensions (``mol_ecp_sphe_agp_jnone``, ``mol_ecp_sphe_j123``,
``mol_ecp_sphe_jdtn``) cover (AO family x geminal x Jastrow x ECP) combinations the
committed ``_rs_fixtures._PATTERNS`` registry doesn't name yet; they are built
directly via ``_build_ham_for_cell`` (the SAME shared builder ``rs_ham`` wraps,
imported read-only from ``_generated_matrix_base``), not invented
from scratch -- verified buildable (see ``test_forces.py``'s module docstring for
the ``_agp_dense_constructible`` / DTN-is-AO-family-independent rationale; the same
holds for an active J3 on the sphe ECP leaf, since ``_J3_KIND["sphe_gto"] == "pgto"``
just like the cart leaf).

Precision regime: the Rust J1/J2 and DTN parameter
gradients (``grad_ln_Psi_jastrow_{1b,2b}_param``, ``grad_ln_Psi_dtn_params``) ignore
the native precision policy and are fp64 in both modes, so their ``jax.grad``
oracles are evaluated with every precision zone at fp64
(``_fp64_reference_policy``) and differentiated with respect to fp64 copies of the
(possibly fp32-stored) parameter leaves, at the same values the Rust kernel reads.
Both steps are no-ops in full mode, so the full-mode references are numerically
unchanged.
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

import contextlib
from functools import partial

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import configure, is_mixed_precision_enabled
from jqmc._precision import get_tolerance_xla_rust_consistency as _backend_consistency_tolerance

# Every comparison in this module is a propagated c-axis aggregate. Keep the
# leaf-kernel suites strict and use the single static mixed aggregate envelope
# here; fp64 resolves to the strict pair.
get_tolerance_xla_rust_consistency = partial(_backend_consistency_tolerance, level="medium")

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

_LOCAL_PATTERNS = {
    "mol_ecp_sphe_agp_jnone": ("water_ccecp_ccpvqz.h5", False, None, "AGP"),
    "mol_ecp_sphe_j123": ("water_ccecp_ccpvqz.h5", True, {"j3": "pgto"}, "SD"),
    "mol_ecp_sphe_jdtn": ("water_ccecp_ccpvqz.h5", True, {"dtn": True}, "SD"),
    # Periodic real-Gamma AGPn (natural-orbital diagonal form) for the lambda_diag map.
    "pbc_agpn_sphepgto": ("lithium.h5", False, None, "AGPn"),
}


@contextlib.contextmanager
def _fp64_reference_policy():
    """Evaluate the enclosed JAX oracle with every precision zone at fp64.

    For a Rust kernel that is fp64 in both modes. In mixed mode the full Python
    policy is activated for the enclosed block and the mixed policy restored
    afterwards (the native Rust policy is not touched); in full mode this is a
    no-op.
    """
    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")


def _fp64_leaves(tree):
    """``tree`` with every real floating leaf widened to float64 (same values).

    ``jax.grad`` returns the cotangent in the primal leaf dtype, so an fp32-stored
    mixed-mode parameter would round the fp64 oracle gradient. A no-op in full mode.
    """

    def widen(x):
        arr = np.asarray(x)
        return arr.astype(np.float64) if arr.dtype.kind == "f" else x

    return jax.tree_util.tree_map(widen, tree)


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


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agp_jnone", "mol_ecp_sphe_agp_jnone"])
def test_grad_ln_det_lambda(pattern):
    """Grad ln det lambda."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    ana = jqmc_rs.grad_ln_det_wrt_lambda(gd, ru, rd)
    ref = jax.grad(lambda g: compute_ln_det_geminal_all_elements(g, ru, rd))(gd).lambda_matrix
    atol, rtol = _backend_consistency_tolerance(("ao_eval", "det_eval"), level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jnone", "mol_ecp_sphe_jnone"])
def test_grad_ln_det_mo(pattern):
    """Grad ln det MO."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    gu, gdn = jqmc_rs.grad_ln_det_wrt_mo_coeff(gd, ru, rd)
    g = jax.grad(lambda G: compute_ln_det_geminal_all_elements(G, ru, rd))(gd)
    atol, rtol = _backend_consistency_tolerance(("ao_eval", "det_eval"), level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(gu), np.asarray(g.orb_data_up_spin.mo_coefficients)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    comparison_actual_2, comparison_reference_2 = np.asarray(gdn), np.asarray(g.orb_data_dn_spin.mo_coefficients)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agpn_jnone", "pbc_agpn_sphepgto"])
def test_grad_ln_det_lambda_diag(pattern):
    """AGPn ``O_k`` on the natural-orbital leaves: the Rust ``grad_ln_det_wrt_lambda_diag``
    (paired-block diagonal + unpaired columns of the dense-``lambda`` gradient) equals the
    ``jax.grad`` of ``ln|det G|`` on ``lambda_diag`` / ``lambda_unpaired``. Molecular GTO and
    real-Gamma sphe-pGTO (the AGPn orbital rides the SAME AO eval as SD)."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    d, u = jqmc_rs.grad_ln_det_wrt_lambda_diag(gd, ru, rd)
    g = jax.grad(lambda G: compute_ln_det_geminal_all_elements(G, ru, rd))(gd)
    atol, rtol = _backend_consistency_tolerance(("ao_eval", "det_eval"), level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(d), np.asarray(g.lambda_diag)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    if np.asarray(gd.lambda_unpaired).size:
        comparison_actual_2, comparison_reference_2 = np.asarray(u), np.asarray(g.lambda_unpaired)
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["pw_gamma_jnone", "pbc_composite_jnone"])
def test_grad_ln_det_mo_pw_composite(pattern):
    """SD ``O_k`` on the mo_coefficients for the pure plane-wave and pGTO(+)PW composite
    leaves (real Gamma). ``grad_ln_det_wrt_mo_coeff`` contracts the full AO stack (empty
    GTO block + PW child for pure PW; gto++pw for the composite) via ``eval_aos``."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    gu, gdn = jqmc_rs.grad_ln_det_wrt_mo_coeff(gd, ru, rd)
    g = jax.grad(lambda G: compute_ln_det_geminal_all_elements(G, ru, rd))(gd)
    atol, rtol = _backend_consistency_tolerance(("ao_eval", "det_eval"), level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(gu), np.asarray(g.orb_data_up_spin.mo_coefficients)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    comparison_actual_2, comparison_reference_2 = np.asarray(gdn), np.asarray(g.orb_data_dn_spin.mo_coefficients)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j12", "mol_ecp_sphe_j12"])
def test_grad_jastrow_12_param(pattern):
    """Grad jastrow 12 param."""
    from jqmc.jastrow_factor import compute_Jastrow_one_body, compute_Jastrow_two_body

    ham = _ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _config(ham)
    j1, j2 = jd.jastrow_one_body_data, jd.jastrow_two_body_data
    atol, rtol = _backend_consistency_tolerance(("jastrow_eval",), level="medium")
    a1 = jqmc_rs.grad_ln_Psi_jastrow_1b_param(j1, ru, rd)
    with _fp64_reference_policy():  # fp64 oracle for the fp64 Rust kernel
        r1 = np.asarray(jax.grad(lambda d: compute_Jastrow_one_body(d, ru, rd))(_fp64_leaves(j1)).jastrow_1b_param)
    comparison_actual_1, comparison_reference_1 = np.asarray(a1), np.asarray(r1)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0
    a2 = jqmc_rs.grad_ln_Psi_jastrow_2b_param(j2, ru, rd)
    with _fp64_reference_policy():
        r2 = np.asarray(jax.grad(lambda d: compute_Jastrow_two_body(d, ru, rd))(_fp64_leaves(j2)).jastrow_2b_param)
    comparison_actual_2, comparison_reference_2 = np.asarray(a2), np.asarray(r2)
    if comparison_actual_2.size or comparison_reference_2.size:
        np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_2.size == comparison_reference_2.size == 0


def _inject_random_j3(ham, seed=2):
    jd = ham.wavefunction_data.jastrow_data
    j3 = jd.jastrow_three_body_data
    n = np.asarray(j3.j_matrix).shape[0]
    jm = np.random.default_rng(seed).normal(scale=0.2, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    return j3.__class__(orb_data=j3.orb_data, j_matrix=jm)


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j123", "mol_ecp_sphe_j123"])
def test_grad_j3_jmatrix(pattern):
    """Grad j3 jmatrix."""
    from jqmc.jastrow_factor import compute_Jastrow_three_body

    ham = _ham(pattern)
    j3 = _inject_random_j3(ham)
    ru, rd = _config(ham)
    ana = jqmc_rs.grad_ln_Psi_jastrow_3b_jmatrix(j3, ru, rd)
    ref = jax.grad(lambda d: compute_Jastrow_three_body(d, ru, rd))(j3).j_matrix
    atol, rtol = _backend_consistency_tolerance(("jastrow_eval",), level="medium")
    comparison_actual_1, comparison_reference_1 = np.asarray(ana), np.asarray(ref)
    if comparison_actual_1.size or comparison_reference_1.size:
        np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
    else:
        assert comparison_actual_1.size == comparison_reference_1.size == 0


def _randomize_dtn(ham, seed=3, scale=0.1):
    """Randomize-then-perturb the DTN parameters (committed jdtn fixtures ship all-zero
    alpha/beta/gamma_indep, which makes a DTN parameter-gradient test vacuous), via the
    dataclass's own ``.replace()``."""
    d = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    rng = np.random.default_rng(seed)
    kw = {}
    for k in ("alpha_indep", "beta_indep", "gamma_indep"):
        v = getattr(d, k, None)
        if v is not None:
            kw[k] = rng.normal(scale=scale, size=np.asarray(v).shape)
    return d.replace(**kw) if kw else d


def _assert_dtn_liveness(d):
    """Guard against the fixture's use_u/use_chi/use_f flags going all-False,
    which would make every ``if d.use_*:`` branch below a silent no-op and the
    whole test vacuously pass with zero blocks actually compared."""
    assert d.use_u or d.use_chi or d.use_f, (
        "DTN fixture has use_u/use_chi/use_f all False -- no block was compared, this test would otherwise pass vacuously"
    )


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jdtn", "mol_ecp_sphe_jdtn"])
def test_grad_dtn_params(pattern):
    """Grad DTN params."""
    from jqmc.jastrow_factor import compute_Jastrow_DTN

    ham = _ham(pattern)
    d = _randomize_dtn(ham)
    ru, rd = _config(ham)
    a, b, g = jqmc_rs.grad_ln_Psi_dtn_params(d, ru, rd)
    with _fp64_reference_policy():  # fp64 oracle for the fp64 Rust kernel
        gref = jax.tree_util.tree_map(np.asarray, jax.grad(lambda dd: compute_Jastrow_DTN(dd, ru, rd))(_fp64_leaves(d)))
    atol, rtol = _backend_consistency_tolerance(("jastrow_eval",), level="medium")
    _assert_dtn_liveness(d)
    if d.use_u:
        comparison_actual_1, comparison_reference_1 = np.asarray(a), np.asarray(gref.alpha_indep)
        if comparison_actual_1.size or comparison_reference_1.size:
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
    if d.use_chi:
        comparison_actual_2, comparison_reference_2 = np.asarray(b), np.asarray(gref.beta_indep)
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0
    if d.use_f:
        comparison_actual_3, comparison_reference_3 = np.asarray(g), np.asarray(gref.gamma_indep)
        if comparison_actual_3.size or comparison_reference_3.size:
            np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_3.size == comparison_reference_3.size == 0


@pytest.mark.heavy
def test_grad_dtn_params_twist():
    """DTN O_k = d(ln Psi)/dc at a COMPLEX twist (k != Gamma). DTN is a REAL
    distance-based Jastrow, so its O_k = dJ_DTN/dc is real and twist-independent --
    the same ``grad_ln_Psi_dtn_params`` (min-image distance) serves it unchanged.
    """
    from jqmc.jastrow_factor import compute_Jastrow_DTN

    ham = rs_ham("twist_sphepgto_jdtn")  # lithium_k025 pbc-ECP twist
    d = _randomize_dtn(ham)
    ru, rd = _config(ham)
    a, b, g = jqmc_rs.grad_ln_Psi_dtn_params(d, ru, rd)
    with _fp64_reference_policy():  # fp64 oracle for the fp64 Rust kernel
        gref = jax.tree_util.tree_map(np.asarray, jax.grad(lambda dd: compute_Jastrow_DTN(dd, ru, rd))(_fp64_leaves(d)))
    atol, rtol = _backend_consistency_tolerance(("jastrow_eval",), level="medium")
    _assert_dtn_liveness(d)
    if d.use_u:
        comparison_actual_1, comparison_reference_1 = np.asarray(a), np.asarray(gref.alpha_indep)
        if comparison_actual_1.size or comparison_reference_1.size:
            np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_1.size == comparison_reference_1.size == 0
    if d.use_chi:
        comparison_actual_2, comparison_reference_2 = np.asarray(b), np.asarray(gref.beta_indep)
        if comparison_actual_2.size or comparison_reference_2.size:
            np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_2.size == comparison_reference_2.size == 0
    if d.use_f:
        comparison_actual_3, comparison_reference_3 = np.asarray(g), np.asarray(gref.gamma_indep)
        if comparison_actual_3.size or comparison_reference_3.size:
            np.testing.assert_allclose(comparison_actual_3, comparison_reference_3, atol=atol, rtol=rtol)
        else:
            assert comparison_actual_3.size == comparison_reference_3.size == 0


# ---- basis-param O_k: d(ln|detG|)/d(radial.{exponents,coefficients}) (cart) ----


# sphe-GTO molecular joins cart (radial sensitivity is angular-kind agnostic:
# same (2l+3)/(4Z) normalization power; only the angular factor P differs). The
# periodic real-pGTO leaf rides the image-folded eval with home-cell wrapping and
# is covered by ``test_grad_radial_basis_periodic`` at the medium (far-image) zone.
@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agp_jnone", "mol_cart_sd_jnone", "mol_sphe_sd_jnone", "mol_sphe_agp_jnone"])
def test_grad_radial_basis(pattern):
    """Grad radial basis."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data

    def _ao(od):
        return od.aos_data if type(od).__name__ == "MOs_data" else od

    atol, rtol = _backend_consistency_tolerance(("ao_eval", "det_eval"), level="medium")
    for seed in range(2):
        ru, rd = _config(ham, seed=seed)
        ana = jqmc_rs.grad_ln_det_wrt_radial(gd, ru, rd)
        g = jax.grad(lambda G: compute_ln_det_geminal_all_elements(G, ru, rd))(gd)
        for side in ("orb_data_up_spin", "orb_data_dn_spin"):
            rad_g = _ao(getattr(g, side)).radial
            aZ, ac = ana[side]
            comparison_actual_1, comparison_reference_1 = np.asarray(aZ), np.asarray(rad_g.exponents)
            if comparison_actual_1.size or comparison_reference_1.size:
                comparison_label = f"{seed}:{side}"  # noqa: F841 - captured by the GDD iteration receipt
                np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_1.size == comparison_reference_1.size == 0
            comparison_actual_2, comparison_reference_2 = np.asarray(ac), np.asarray(rad_g.coefficients)
            if comparison_actual_2.size or comparison_reference_2.size:
                comparison_label = f"{seed}:{side}"  # noqa: F841 - captured by the GDD iteration receipt
                np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_2.size == comparison_reference_2.size == 0


# ---- basis-param O_k on the PERIODIC real-pGTO leaf (image-folded, home-cell
#      wrapped). Medium zone: the far-image radial cutoff / min-image differ from
#      XLA's static full-image sum at the ~1e-6 level (pGTO far-image tail). ----


@pytest.mark.light
def test_grad_radial_basis_periodic():
    """Grad radial basis periodic."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham("pbc_gamma_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data

    def _ao(od):
        return od.aos_data if type(od).__name__ == "MOs_data" else od

    atol, rtol = _backend_consistency_tolerance(("ao_eval", "det_eval"), level="medium")
    for seed in range(2):
        ru, rd = _config(ham, seed=seed)
        ana = jqmc_rs.grad_ln_det_wrt_radial(gd, ru, rd)
        g = jax.grad(lambda G: compute_ln_det_geminal_all_elements(G, ru, rd))(gd)
        for side in ("orb_data_up_spin", "orb_data_dn_spin"):
            rad_g = _ao(getattr(g, side)).radial
            aZ, ac = ana[side]
            comparison_actual_1, comparison_reference_1 = np.asarray(aZ), np.asarray(rad_g.exponents)
            if comparison_actual_1.size or comparison_reference_1.size:
                comparison_label = f"{seed}:{side}"  # noqa: F841 - captured by the GDD iteration receipt
                np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_1.size == comparison_reference_1.size == 0
            comparison_actual_2, comparison_reference_2 = np.asarray(ac), np.asarray(rad_g.coefficients)
            if comparison_actual_2.size or comparison_reference_2.size:
                comparison_label = f"{seed}:{side}"  # noqa: F841 - captured by the GDD iteration receipt
                np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_2.size == comparison_reference_2.size == 0


# ---- basis-param: d(T_det)/d(radial.{exponents,coefficients}) (jnone kinetic; cart + sphe) ----


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agp_jnone", "mol_cart_sd_jnone", "mol_sphe_sd_jnone", "mol_sphe_agp_jnone"])
def test_grad_radial_kinetic(pattern):
    """Grad radial kinetic."""
    from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data.replace(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
    )

    def _ao(od):
        return od.aos_data if type(od).__name__ == "MOs_data" else od

    atol, rtol = _backend_consistency_tolerance(
        ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic"), level="medium"
    )
    for seed in range(2):
        ru, rd = _config(ham, seed=seed)
        ana = jqmc_rs.grad_eL_wrt_radial(gd, ru, rd)
        g = jax.grad(lambda G: compute_kinetic_energy(Wavefunction_data(jastrow_data=jd, geminal_data=G), ru, rd))(gd)
        for side in ("orb_data_up_spin", "orb_data_dn_spin"):
            rad_g = _ao(getattr(g, side)).radial
            aZ, ac = ana[side]
            comparison_actual_1, comparison_reference_1 = np.asarray(aZ), np.asarray(rad_g.exponents)
            if comparison_actual_1.size or comparison_reference_1.size:
                comparison_label = f"{seed}:{side}"
                np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_1.size == comparison_reference_1.size == 0
            comparison_actual_2, comparison_reference_2 = np.asarray(ac), np.asarray(rad_g.coefficients)
            if comparison_actual_2.size or comparison_reference_2.size:
                comparison_label = f"{seed}:{side}"
                np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_2.size == comparison_reference_2.size == 0


# ---- basis-param d(T_det)/d(radial) on the PERIODIC real-pGTO leaf (medium zone). ----


@pytest.mark.light
def test_grad_radial_kinetic_periodic():
    """Grad radial kinetic periodic."""
    from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy

    ham = _ham("pbc_gamma_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data.replace(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
    )

    def _ao(od):
        return od.aos_data if type(od).__name__ == "MOs_data" else od

    atol, rtol = _backend_consistency_tolerance(
        ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic"), level="medium"
    )
    for seed in range(2):
        ru, rd = _config(ham, seed=seed)
        ana = jqmc_rs.grad_eL_wrt_radial(gd, ru, rd)
        g = jax.grad(lambda G: compute_kinetic_energy(Wavefunction_data(jastrow_data=jd, geminal_data=G), ru, rd))(gd)
        for side in ("orb_data_up_spin", "orb_data_dn_spin"):
            rad_g = _ao(getattr(g, side)).radial
            aZ, ac = ana[side]
            comparison_actual_1, comparison_reference_1 = np.asarray(aZ), np.asarray(rad_g.exponents)
            if comparison_actual_1.size or comparison_reference_1.size:
                comparison_label = f"{seed}:{side}"
                np.testing.assert_allclose(comparison_actual_1, comparison_reference_1, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_1.size == comparison_reference_1.size == 0
            comparison_actual_2, comparison_reference_2 = np.asarray(ac), np.asarray(rad_g.coefficients)
            if comparison_actual_2.size or comparison_reference_2.size:
                comparison_label = f"{seed}:{side}"
                np.testing.assert_allclose(comparison_actual_2, comparison_reference_2, atol=atol, rtol=rtol)
            else:
                assert comparison_actual_2.size == comparison_reference_2.size == 0

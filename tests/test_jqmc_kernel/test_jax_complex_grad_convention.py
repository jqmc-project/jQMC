"""Wirtinger / complex-gradient convention guard tests.

These tests pin JAX's complex-autodiff convention that complex wavefunctions
rely on. The single load-bearing fact is:

    for a *real-valued* loss ``f: C -> R``, ``jax.grad(f)(z) == 2 * conj(...)``
    direction, i.e. ``grad(|z|**2)(z0) == 2 * conj(z0)``.

This is why the stored generalized-force "``O_matrix``" holds ``conj(O_k)``
(JAX ``grad`` of the real ``ln|Psi|`` w.r.t. a complex parameter), so the
SR/force code conjugates at the aggregation boundary rather than in
the grad wrappers. If a future JAX upgrade silently changes this
convention, these tests fail loudly before any physics test does.
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

from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_COMPLEX_GRAD_FD, RTOL_NONVACUITY_DISTINCTION
from jqmc._precision import get_tolerance, get_tolerance_min
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# The complex (twist) path stores complex128 wavefunction parameters; the convention only holds
# meaningfully in double precision (the production config also enables x64).
jax.config.update("jax_enable_x64", True)

# Imports below are used by the empirical complex-gradient verification.
from jqmc._setting import ATOL_COMPLEX_GRAD_FD, ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT
from jqmc.atomic_orbital import _apply_image_expansion, _promote_pgto_to_complex
from jqmc.determinant import Geminal_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel.test_atomic_orbital import _make_aos_data


class TestJaxComplexGradConvention:
    @pytest.mark.light
    def test_grad_abs_sq_is_two_conj(self):
        """``jax.grad(|c|**2)(c0) == 2 * conj(c0)`` (JAX's real-loss convention)."""
        c0 = jnp.array(1.0 + 2.0j, dtype=jnp.complex128)
        g = jax.grad(lambda c: jnp.abs(c) ** 2)(c0)
        assert g.dtype == jnp.complex128
        np.testing.assert_allclose(g, 2.0 * jnp.conj(c0), rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        # And NOT the un-conjugated direction, so a convention flip is caught.
        assert not jnp.allclose(g, 2.0 * c0, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)

    @pytest.mark.parametrize(
        "c0",
        [
            0.7 + 0.0j,
            0.0 + 1.3j,
            -2.1 + 0.9j,
            3.0 - 4.0j,
        ],
    )
    @pytest.mark.light
    def test_grad_abs_sq_is_two_conj_param(self, c0):
        """Same identity across several complex points incl. the real/imag axes."""
        z = jnp.array(c0, dtype=jnp.complex128)
        g = jax.grad(lambda c: jnp.abs(c) ** 2)(z)
        np.testing.assert_allclose(g, 2.0 * jnp.conj(z), rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_grad_real_part_of_holomorphic_log_is_unconjugated(self):
        """``grad(Re ln(c))(c0) == 1/c0`` (the UN-conjugated holomorphic derivative).

        JAX's real-loss convention is ``grad(f)(c) = 2 * d f / d c`` (Wirtinger
        derivative w.r.t. ``c``, not ``c-bar``). When the integrand is *holomorphic*
        in ``c`` -- as ``ln(c)`` is, and as ``ln Psi`` is in the determinant
        parameters taken alone -- the conjugate part drops and the gradient is the
        plain holomorphic derivative ``O = d ln Psi / dc``, NOT ``conj(O)``.

        This is the scalar analogue of ``grad(ln|Psi|)`` (``ln|Psi| = Re ln Psi``).
        Contrast with :func:`test_grad_abs_sq_is_two_conj`, where the loss ``|c|**2``
        is *not* of the form ``Re(holomorphic)`` and the gradient carries a conj.

        NOTE: the stored-gradient convention says the ``O_matrix``
        holds ``conj(O_k)``. That conjugation does NOT come from this bare-log rule
        -- it can only arise from the non-holomorphic structure of the actual stored
        parametrization (Convention-B down-spin = ``conj(C)``, Hermitian paired
        ``lambda``). The convention must therefore be pinned against the real
        ``evaluate_ln_wavefunction``, not assumed from this scalar test.
        """
        c0 = jnp.array(1.5 - 0.5j, dtype=jnp.complex128)
        g = jax.grad(lambda c: jnp.real(jnp.log(c)))(c0)
        np.testing.assert_allclose(g, 1.0 / c0, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        # The "2 * d/dc" form, stated explicitly so a convention flip is loud.
        np.testing.assert_allclose(g, 2.0 * (0.5 / c0), rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_grad_abs_sq_matches_real_gradient(self):
        """The complex grad packs the real 2D gradient: ``grad = df/dx + i df/dy``.

        For ``f(z) = |z|**2 = x**2 + y**2`` the real gradient is ``(2x, 2y)`` and
        JAX returns ``2x - 2i y = 2 conj(z)`` (note the sign on the imaginary part
        is what distinguishes the convention).
        """
        x0, y0 = 1.1, -2.3
        z0 = jnp.array(x0 + 1j * y0, dtype=jnp.complex128)
        g = jax.grad(lambda c: jnp.abs(c) ** 2)(z0)
        np.testing.assert_allclose(jnp.real(g), 2.0 * x0, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(jnp.imag(g), -2.0 * y0, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)


# ---------------------------------------------------------------------------
# Empirical verification of the complex-gradient convention.
# Pins the empirical complex-gradient convention:
#   ``jax.grad(ln|Psi|, lambda)`` returns ``O_k`` (un-conjugated), NOT ``conj(O_k)``.
# ---------------------------------------------------------------------------


def _build_complex_fixture(seed: int = 1234, n_up: int = 2):
    """Build a non-trivial complex Bloch-pGTO wavefunction at non-Gamma."""
    cell = CELL_FIXTURES["cubic_10bohr"]
    positions = np.array([[3.0, 3.0, 3.0]], dtype=np.float64)
    s = make_pbc_structure(
        cell,
        positions=positions,
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )
    k_point = (0.3, 0.15, -0.07)
    base = _make_aos_data(s, "cart", l_max=1, uniform=True)
    real_pgto = _apply_image_expansion(base, s)
    cplx_leaf = _promote_pgto_to_complex(real_pgto, k_point)

    rng = np.random.default_rng(seed)
    C = (rng.normal(size=(n_up, cplx_leaf.num_ao)) + 1j * rng.normal(size=(n_up, cplx_leaf.num_ao))).astype(np.complex128)
    mos = MOs_data(num_mo=n_up, aos_data=cplx_leaf, mo_coefficients=C)
    # BCS / TR-pair: dn channel uses the -k AO leaf with conj(C).
    cplx_leaf_dn = cplx_leaf.with_negated_k()
    mos_dn = MOs_data(num_mo=n_up, aos_data=cplx_leaf_dn, mo_coefficients=np.conj(C))
    A = (rng.normal(size=(n_up, n_up)) + 1j * rng.normal(size=(n_up, n_up))).astype(np.complex128)
    lam = 0.5 * (A + A.conj().T)
    gem = Geminal_data.build(
        num_electron_up=n_up,
        num_electron_dn=n_up,
        orb_data_up_spin=mos,
        orb_data_dn_spin=mos_dn,
        lambda_matrix=lam,
    )
    jas = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)
    wf = Wavefunction_data(geminal_data=gem, jastrow_data=jas)
    r_up = rng.uniform(2.0, 8.0, size=(n_up, 3)).astype(np.float64)
    r_dn = rng.uniform(2.0, 8.0, size=(n_up, 3)).astype(np.float64)
    return wf, gem, lam, jnp.asarray(r_up), jnp.asarray(r_dn)


# Zones whose arithmetic enters d ln|Psi| / d lambda on this Jastrow-free
# fixture. The orbital tables (ao_eval, mo_eval) do not depend on lambda: AD and
# FD both differentiate the same map lambda -> ln|det(U.T @ lambda @ D)| with
# the identical orbital tables of the active precision mode, so the derivative
# and the FD difference quotient are formed in the geminal/log-det and
# ln|Psi|-assembly zones only.
_LAMBDA_DERIVATIVE_ZONES = ("det_eval", "wf_eval")


def _finite_diff_O(eval_lam, lam, i, j, h: float = 1e-6):
    """Finite-difference ``2 * d (Re ln Psi) / d lambda_{ij}`` (Wirtinger holomorphic).

    JAX convention: ``grad(f: C->R)(c) = d/d Re(c) - i * d/d Im(c) = 2 d/dc``.
    """
    lam_pl_re = lam.copy()
    lam_pl_re[i, j] += h
    lam_mn_re = lam.copy()
    lam_mn_re[i, j] -= h
    lam_pl_im = lam.copy()
    lam_pl_im[i, j] += 1j * h
    lam_mn_im = lam.copy()
    lam_mn_im[i, j] -= 1j * h
    dRe = (eval_lam(lam_pl_re) - eval_lam(lam_mn_re)) / (2 * h)
    dIm = (eval_lam(lam_pl_im) - eval_lam(lam_mn_im)) / (2 * h)
    return dRe - 1j * dIm


class TestPhase3JAXGradConvention:
    @pytest.mark.light
    def test_real_wavefunction_lambda_gradient_matches_inverse_reference(self):
        """The original Bloch-pGTO/Hermitian-lambda fixture, without FD or AD in the reference.

        For M = U.T @ lambda @ D, d log(det M)/d lambda =
        U @ inv(M).T @ D.T.  The down-spin orbital already contains its
        prescribed conjugation; adding a conjugation to this result is wrong.
        Both orbital tables are evaluated in the active precision mode.
        """
        wf, gem, lam, r_up, r_dn = _build_complex_fixture()
        actual = np.asarray(jax.grad(evaluate_ln_wavefunction, argnums=0)(wf, r_up, r_dn).geminal_data.lambda_matrix)
        up = np.asarray(gem.compute_orb_api(gem.orb_data_up_spin, r_up), dtype=np.complex128)
        dn = np.asarray(gem.compute_orb_api(gem.orb_data_dn_spin, r_dn), dtype=np.complex128)
        matrix = up.T @ lam @ dn
        reference = up @ np.linalg.inv(matrix).T @ dn.T
        atol, rtol = get_tolerance("det_eval", "strict")
        assert actual.dtype == np.complex128
        assert actual.shape == lam.shape
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
        off_diagonal = ~np.eye(lam.shape[0], dtype=bool)
        assert np.any(np.abs(reference[off_diagonal].imag) > ATOL_NONVACUITY_DISTINCTION)
        assert not np.allclose(
            actual[off_diagonal],
            reference[off_diagonal].conj(),
            atol=atol,
            rtol=rtol,
        ), "The conjugated inverse reference must be distinguishable on the original fixture."

    @pytest.mark.light
    def test_jax_grad_returns_O_not_conj_O(self):
        """The anchor: ``jax.grad(ln|Psi|, lambda)`` stores ``O_k`` un-conjugated.

        A regression here means the JAX convention has shifted -- the
        SR/LM ``T -> conj().T`` mechanical edits remain mathematically correct
        in either convention, but the INTERPRETATION of the stored-gradient
        convention must be re-derived (otherwise the conj placement in ``f`` /
        ``solve_linear_method`` / ``get_aF`` will be off by one).
        """
        wf, gem, lam, r_up, r_dn = _build_complex_fixture()
        grad_wf = jax.grad(evaluate_ln_wavefunction, argnums=0)(wf, r_up, r_dn)
        G = np.asarray(grad_wf.geminal_data.lambda_matrix)
        assert G.dtype == np.complex128
        assert G.shape == lam.shape

        def eval_lam(new_lam):
            g2 = gem.replace(lambda_matrix=np.ascontiguousarray(new_lam, dtype=np.complex128))
            w2 = wf.replace(geminal_data=g2)
            return float(evaluate_ln_wavefunction(w2, r_up, r_dn))

        # The off-diagonal entries of a Hermitian lambda have non-trivial Im
        # parts -- these are the entries that distinguish ``O`` from ``conj(O)``.
        # Diagonal entries of a Hermitian lambda are real, so ``O == conj(O)``
        # trivially there and they cannot tell the two conventions apart.
        n = G.shape[0]
        off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
        assert len(off_diag) >= 1, "Need an off-diagonal entry to distinguish O from conj(O)."
        # Each mode is judged within its own precision: the FD truncation
        # budget is combined with the central zone budget of the zones the
        # derivative crosses in the ACTIVE mode. Both G and the FD reference
        # come from the same active-mode evaluate_ln_wavefunction; nothing is
        # compared against another precision mode.
        zone_atol, zone_rtol = get_tolerance_min(_LAMBDA_DERIVATIVE_ZONES, "strict")
        fd_atol = max(ATOL_COMPLEX_GRAD_FD, zone_atol)
        fd_rtol = max(RTOL_COMPLEX_GRAD_FD, zone_rtol)
        matches_O = 0
        matches_conj_O = 0
        for i, j in off_diag:
            fd = _finite_diff_O(eval_lam, np.asarray(lam), i, j)
            jg = G[i, j]
            # Each entry must match ONE of the two conventions to fp-FD tolerance.
            is_O = np.isclose(jg, fd, atol=fd_atol, rtol=fd_rtol)
            is_conj_O = np.isclose(jg, np.conj(fd), atol=fd_atol, rtol=fd_rtol)
            matches_O += int(is_O)
            matches_conj_O += int(is_conj_O)
            # On a generic non-real Hermitian entry, the two cannot both match.
            if abs(fd.imag) > 1e-5:
                assert is_O ^ is_conj_O, f"Entry ({i},{j}): FD={fd}, JAX={jg}; cannot tell O from conj(O) -- check h."
        assert matches_O == len(off_diag), (
            f"jax.grad does NOT match O across all off-diagonal entries "
            f"({matches_O}/{len(off_diag)} match O, {matches_conj_O}/{len(off_diag)} match conj(O)). "
            "If this fails, JAX's complex-grad convention has shifted; the stored-gradient "
            "INTERPRETATION must be re-derived (but the mechanical conj().T edits stay the same)."
        )
        # And conj(O) does NOT match in general (on entries with nonzero Im).
        assert matches_conj_O < len(off_diag), (
            "JAX grad matches BOTH O and conj(O) on every off-diagonal entry; "
            "the fixture is degenerate -- pick a less symmetric lambda / r_carts."
        )

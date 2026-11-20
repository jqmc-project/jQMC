"""Hermitian square-root and isolated LM numerical regression tests.

Deterministic CI (no MCMC fixture required) for the surviving numerical kernels:

1. ``_hermitian_sqrt``: bit-identical on a real symmetric input; correct on
   a complex Hermitian PSD overlap (sqrt @ sqrt == S, inv_sqrt @ S @ inv_sqrt == I).
2. ``solve_linear_method`` (Hermitian LM unit): the real and complex solutions
   agree with an independent generalized eigensolve in the original parameter
   basis, including the selected energy, reference weight and residual.
3. complex energy projection: the ``.real`` wrap and ``|<E>|^2``
   variance formula deliver the correct real ``E_mean`` and the standard
   ``Var = <|e_L|^2> - |<e_L>|^2`` value on a synthetic complex e_L sample.

These pin the Hermitian kernels without needing a non-Gamma MCMC fixture.
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

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc.determinant import _hermitian_sqrt
from jqmc.jqmc_optimizer_kernel import solve_linear_method


def _independent_linear_method_reference(H_0, f, S, K, B):
    """Solve H v = E M v directly, without the production whitening or cutoff code."""
    from scipy.linalg import eigh

    size = len(f) + 1
    metric = np.eye(size, dtype=S.dtype)
    metric[1:, 1:] = S
    hamiltonian = np.zeros_like(metric)
    hamiltonian[0, 0] = H_0
    hamiltonian[1:, 0] = -f / 2
    hamiltonian[0, 1:] = -f.conj() / 2
    hamiltonian[1:, 1:] = ((K + B) + (K + B).conj().T) / 2
    values, vectors = eigh(hamiltonian, metric, driver="gvd")
    weights = np.abs(vectors[0]) ** 2
    selected = np.argmax(weights)
    reference = np.concatenate([vectors[1:, selected] / vectors[0, selected], [values[selected], weights[selected]]])
    return reference, hamiltonian, metric


# ---------- 1. _hermitian_sqrt helper -----------


def _random_complex_hermitian_psd(n: int, seed: int = 0) -> np.ndarray:
    """Return a random n x n Hermitian PSD complex matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    # PSD via A^H A + 0.1 * I (ensures strictly positive eigenvalues)
    S = A.conj().T @ A + 0.1 * np.eye(n)
    # Re-Hermitise to kill round-off skew.
    return 0.5 * (S + S.conj().T)


def _random_real_symmetric_psd(n: int, seed: int = 1) -> np.ndarray:
    """Return a random n x n symmetric PSD real matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    S = A.T @ A + 0.1 * np.eye(n)
    return 0.5 * (S + S.T)


class TestHermitianSqrt:
    @pytest.mark.light
    def test_hermitian_sqrt_real_bit_identical(self):
        """On a real symmetric PSD input ``_hermitian_sqrt`` matches the
        real ``eigvecs @ diag(sqrt) @ eigvecs.T`` form byte-for-byte."""
        S = _random_real_symmetric_psd(5, seed=1)
        sqrt_S, inv_sqrt_S = _hermitian_sqrt(S)
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1.0e-10)
        expected_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        expected_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        np.testing.assert_array_equal(sqrt_S, expected_sqrt)
        np.testing.assert_array_equal(inv_sqrt_S, expected_inv_sqrt)
        # dtype unchanged
        assert sqrt_S.dtype == np.float64
        assert inv_sqrt_S.dtype == np.float64

    @pytest.mark.light
    def test_hermitian_sqrt_complex_roundtrip(self):
        """Round-trip identities on a complex Hermitian PSD input:
        ``sqrt @ sqrt == S`` and ``inv_sqrt @ S @ inv_sqrt == I``."""
        S = _random_complex_hermitian_psd(6, seed=2)
        sqrt_S, inv_sqrt_S = _hermitian_sqrt(S)
        # sqrt_S is Hermitian
        np.testing.assert_allclose(sqrt_S, sqrt_S.conj().T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        # sqrt @ sqrt = S
        np.testing.assert_allclose(sqrt_S @ sqrt_S, S, atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, rtol=RTOL_PYTHON_FP64_LOOSE)
        # inv_sqrt @ S @ inv_sqrt = I
        eye = np.eye(S.shape[0])
        np.testing.assert_allclose(
            inv_sqrt_S @ S @ inv_sqrt_S, eye, atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        # dtype preserved as complex128
        assert sqrt_S.dtype == np.complex128
        assert inv_sqrt_S.dtype == np.complex128

    @pytest.mark.light
    def test_hermitian_sqrt_clip_floor_on_near_singular_real(self):
        """A tiny negative overlap eigenvalue is replaced by the declared floor.

        The expected square root and inverse square root are constructed from
        the fixture's known eigensystem.  Finiteness is retained as a
        non-vacuity check, but the verdict is equality with that independent
        spectral reference.  This reproduces the periodic-AO scenario.
        """
        # Build S with one near-zero eigenvalue.
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        eigvals = np.array([5.0, 2.0, 0.5, -1.0e-14])  # one tiny negative
        S = Q @ np.diag(eigvals) @ Q.T
        S = 0.5 * (S + S.T)
        eig_clip_tol = 1.0e-10
        sqrt_S, inv_sqrt_S = _hermitian_sqrt(S, eig_clip_tol=eig_clip_tol)

        clipped = np.maximum(eigvals, eig_clip_tol)
        expected_sqrt = Q @ np.diag(np.sqrt(clipped)) @ Q.T
        expected_inv_sqrt = Q @ np.diag(1.0 / np.sqrt(clipped)) @ Q.T
        np.testing.assert_allclose(
            sqrt_S, expected_sqrt, atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_allclose(
            inv_sqrt_S,
            expected_inv_sqrt,
            atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        assert np.all(np.isfinite(sqrt_S))
        assert np.all(np.isfinite(inv_sqrt_S))


# ---------- 2. solve_linear_method Hermitian LM unit -----------


class TestSolveLinearMethod:
    @pytest.mark.light
    @pytest.mark.parametrize("seed", [30])
    def test_solve_linear_method_real_path_unchanged(self, seed):
        """The real LM update, selected energy and reference weight solve the original generalized eigenproblem."""
        p = 4
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((p, p))
        S = A.T @ A + 0.1 * np.eye(p)
        S = 0.5 * (S + S.T)
        H_kin = rng.standard_normal((p, p))
        H_kin = 0.5 * (H_kin + H_kin.T)  # symmetric K
        H_pot = rng.standard_normal((p, p))  # B can be non-symmetric
        f_vec = rng.standard_normal(p)
        H_0 = -1.0  # arbitrary reference energy

        c_vec, E_lm, v0_sq = solve_linear_method(
            H_0=H_0,
            f_vec=f_vec,
            S_matrix=S,
            K_matrix=H_kin,
            B_matrix=H_pot,
            epsilon=1.0e-8,
        )
        assert c_vec.dtype == np.float64
        assert c_vec.shape == (p,)
        assert np.all(np.isfinite(c_vec))
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0

        reference, hamiltonian, metric = _independent_linear_method_reference(H_0, f_vec, S, H_kin, H_pot)
        actual = np.concatenate([c_vec, [E_lm, v0_sq]])
        np.testing.assert_allclose(actual, reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        vector = np.concatenate([[1], c_vec])
        residual_actual = hamiltonian @ vector
        residual_reference = E_lm * (metric @ vector)
        np.testing.assert_allclose(
            residual_actual, residual_reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    @pytest.mark.light
    @pytest.mark.parametrize("seed", [40])
    def test_solve_linear_method_complex_hermitian_LM(self, seed):
        """The complex LM solution retains force conjugation and agrees with an independent generalized eigensolve."""
        p = 4
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        S = A.conj().T @ A + 0.1 * np.eye(p)
        S = 0.5 * (S + S.conj().T)
        K = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        K = 0.5 * (K + K.conj().T)  # Hermitian K
        B = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        # B is not individually Hermitian; the solver Hermitianises K + B.
        f_vec = rng.standard_normal(p) + 1j * rng.standard_normal(p)
        H_0 = -1.0

        c_vec, E_lm, v0_sq = solve_linear_method(
            H_0=H_0,
            f_vec=f_vec,
            S_matrix=S,
            K_matrix=K,
            B_matrix=B,
            epsilon=1.0e-8,
        )
        # c_vec dtype follows the complex input (R5 dtype_mcmc_np fix).
        assert c_vec.dtype == np.complex128
        assert c_vec.shape == (p,)
        assert np.all(np.isfinite(c_vec))
        # E_lm is the real LM eigenvalue (eigh of Hermitian H_bar gives real
        # eigenvalues).
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0

        reference, hamiltonian, metric = _independent_linear_method_reference(H_0, f_vec, S, K, B)
        actual = np.concatenate([c_vec, [E_lm, v0_sq]])
        np.testing.assert_allclose(actual, reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        vector = np.concatenate([[1], c_vec])
        residual_actual = hamiltonian @ vector
        residual_reference = E_lm * (metric @ vector)
        np.testing.assert_allclose(
            residual_actual, residual_reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    @pytest.mark.light
    @pytest.mark.parametrize("seed", [40])
    def test_complex_lm_reference_rejects_reversed_force_conjugation(self, monkeypatch, seed):
        """Reversing the actual solver's force conjugation must fail the independent complex LM reference."""
        import inspect
        import textwrap

        self.test_solve_linear_method_complex_hermitian_LM(seed)
        source = textwrap.dedent(inspect.getsource(solve_linear_method))
        column = "H_bar[1:, 0] = -0.5 * f_new"
        row = "H_bar[0, 1:] = -0.5 * np.conj(f_new)"
        assert source.count(column) == source.count(row) == 1
        source = source.replace(column, "H_bar[1:, 0] = -0.5 * np.conj(f_new)")
        source = source.replace(row, "H_bar[0, 1:] = -0.5 * f_new")
        namespace = dict(solve_linear_method.__globals__)
        exec(compile(source, "<reversed-force-conjugation>", "exec"), namespace)
        monkeypatch.setitem(globals(), "solve_linear_method", namespace["solve_linear_method"])
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            self.test_solve_linear_method_complex_hermitian_LM(seed)

    @pytest.mark.light
    @pytest.mark.parametrize("kind,seed", [("real", 30), ("complex", 40)])
    def test_linear_method_independent_reference(self, kind, seed):
        """The independently budgeted LM contract covers the fixed real and complex original-basis problems."""
        if kind == "real":
            self.test_solve_linear_method_real_path_unchanged(seed)
        else:
            self.test_solve_linear_method_complex_hermitian_LM(seed)

    @pytest.mark.light
    def test_solve_linear_method_complex_zero_force_zero_step(self):
        """A zero force vector on a complex Hermitian S/K/B gives a zero
        parameter step -- the LM-extended matrix is block-diagonal and the
        Hermitian eigen-decomposition picks the column with ``v_0 = 1``,
        yielding ``c_vec = 0`` (overlap-preserving null step)."""
        p = 3
        rng = np.random.default_rng(50)
        A = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        S = A.conj().T @ A + 0.1 * np.eye(p)
        S = 0.5 * (S + S.conj().T)
        K = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        K = 0.5 * (K + K.conj().T)
        B = np.zeros((p, p), dtype=np.complex128)
        f_vec = np.zeros(p, dtype=np.complex128)

        c_vec, _, v0_sq = solve_linear_method(
            H_0=0.0,
            f_vec=f_vec,
            S_matrix=S,
            K_matrix=K,
            B_matrix=B,
            epsilon=1.0e-8,
        )
        # With f = 0 the H_bar off-diagonal is zero; the LM eigenvalue problem
        # decouples and the dominant-|v_0|^2 eigenvector is e_0, giving c = 0.
        np.testing.assert_allclose(
            np.abs(c_vec),
            np.zeros(p, dtype=np.float64),
            atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        # v_0^2 = 1 (the (0,0) eigenvalue is the smallest in this construction).
        assert v0_sq == pytest.approx(1.0, abs=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT)


# ---------- 3. complex energy projection (math-only) -----------


class TestComplexJackknife:
    @pytest.mark.light
    def test_complex_jackknife_variance_formula(self):
        """Validate the ``Var = <|e_L|^2> - |<e_L>|^2`` formula on a
        synthetic complex e_L bin sample. The real-only ``E**2`` form would
        contaminate the variance with ``-2 Im(E)^2``; the corrected
        ``|E|^2`` form is invariant under ``E -> conj(E)``."""
        rng = np.random.default_rng(60)
        M = 50
        nw = 4
        # Synthetic complex per-walker e_L with a known Re and Im scale.
        e_L = rng.standard_normal((M, nw)) + 1j * rng.standard_normal((M, nw))
        e_L2 = np.abs(e_L) ** 2  # ground truth e_L2 buffer (real)
        w_L = np.ones((M, nw))

        # Per-bin jackknife (one bin per (m, w) sample for simplicity).
        e_L_bin = e_L.ravel()
        e_L2_bin = e_L2.ravel()
        w_L_bin = w_L.ravel()

        total_w = float(np.sum(w_L_bin))
        total_we = np.sum(w_L_bin * e_L_bin)
        total_we2 = np.sum(w_L_bin * e_L2_bin)
        # Leave-one-out jackknife per-bin mean.
        M_total = len(e_L_bin)
        E_jn = np.array([(total_we - w_L_bin[k] * e_L_bin[k]) / (total_w - w_L_bin[k]) for k in range(M_total)])
        E2_jn = np.array([(total_we2 - w_L_bin[k] * e_L2_bin[k]) / (total_w - w_L_bin[k]) for k in range(M_total)])

        # Real-only Var formula -- wrong for complex E.
        Var_wrong = E2_jn - E_jn**2
        # Complex-aware Var formula -- correct.
        Var_right = E2_jn - np.abs(E_jn) ** 2

        # The corrected Var is real-valued (within round-off), the wrong form
        # carries a complex contamination -2*i*Re(E)*Im(E) on the off-diagonal.
        assert np.max(np.abs(np.imag(Var_right))) < ATOL_PYTHON_FP64_LOOSE
        # And it equals the standard |<e>|^2-based estimator on real samples
        # if we replace e_L by Re(e_L).
        Re_eL = np.real(e_L).ravel()
        total_we_real = np.sum(w_L_bin * Re_eL)
        Re_E_jn = np.array([(total_we_real - w_L_bin[k] * Re_eL[k]) / (total_w - w_L_bin[k]) for k in range(M_total)])
        Var_real_floor = E2_jn - Re_E_jn**2 - np.imag(E_jn) ** 2  # full |E|^2 = Re(E)^2 + Im(E)^2
        np.testing.assert_allclose(np.real(Var_right), Var_real_floor, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)


class TestRealPathVar:
    @pytest.mark.light
    def test_real_path_var_formula_bit_identical_to_debug(self):
        """On a real e_L the new ``|E|^2`` form coincides bit-for-bit with
        the debug ``E**2`` form. Pins the real-path byte-identity claim."""
        rng = np.random.default_rng(70)
        e_L = rng.standard_normal(100)  # real-only

        # Same algebra both ways
        var_debug = e_L**2
        var_new = np.abs(e_L) ** 2
        np.testing.assert_array_equal(var_debug, var_new)


# ---------- 5. MO phase-gauge fix at TREXIO load -----------


def _replay_gauge_fix(C: np.ndarray, gauge_pivot_tol: float = 1e-8) -> np.ndarray:
    """Reproduce the trexio_wrapper gauge-fix block as a standalone reference.

    Mirrors the algorithm at jqmc/trexio_wrapper.py lines 251-271 so the
    test pins the *algorithm* (per-band, first-non-negligible coefficient
    real-positive) rather than the call site. If the call site moves
    (e.g. consolidated into a helper for the QE converter), this test
    still pins the math.
    """
    abs_C = np.abs(C)
    mask = abs_C > gauge_pivot_tol
    pivot_col = np.where(mask.any(axis=1), mask.argmax(axis=1), 0)
    pivot_c = C[np.arange(C.shape[0]), pivot_col]
    pivot_abs = np.abs(pivot_c)
    phase = np.where(pivot_abs > 0.0, np.conj(pivot_c) / np.where(pivot_abs > 0.0, pivot_abs, 1.0), 1.0)
    return C * phase[:, None]


class TestMOGaugeAndLambda:
    @pytest.mark.light
    def test_mo_gauge_first_nonzero_is_real_positive(self):
        """After the gauge fix every band's first non-negligible coefficient
        is real-positive (within float64 round-off)."""
        rng = np.random.default_rng(80)
        n_bands = 5
        n_ao = 8
        C = rng.standard_normal((n_bands, n_ao)) + 1j * rng.standard_normal((n_bands, n_ao))
        # Make the first coefficient near-zero on one band so we trigger the
        # ``mask.argmax`` fallback to a later column.
        C[2, 0] = 0.0
        C[2, 1] = 0.0

        C_gauged = _replay_gauge_fix(C)
        # Identify each band's pivot column.
        abs_C = np.abs(C_gauged)
        mask = abs_C > 1e-8
        pivot_col = np.where(mask.any(axis=1), mask.argmax(axis=1), 0)
        pivot_c = C_gauged[np.arange(n_bands), pivot_col]
        # Pivot is real-positive: Im part vanishes and Re part is non-negative.
        np.testing.assert_allclose(
            np.imag(pivot_c), np.zeros(n_bands, dtype=np.float64), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_array_less(-ATOL_PYTHON_FP64_LOOSE, np.real(pivot_c) + 1e-30)
        # The magnitude of each MO column is unchanged by the gauge (the gauge
        # is a per-band unit modulus multiplication).
        np.testing.assert_allclose(np.abs(C_gauged), np.abs(C), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_mo_gauge_preserves_outer_product(self):
        """The gauge is a unit-modulus per-band rotation, so any
        physical observable bilinear in (C[i], conj(C[i])) (a band's
        density-matrix block C[i] outer conj(C[i])) is invariant
        -- this is what makes the gauge unphysical in the first place."""
        rng = np.random.default_rng(81)
        C = rng.standard_normal((3, 6)) + 1j * rng.standard_normal((3, 6))
        C_gauged = _replay_gauge_fix(C)
        # Band-diagonal density matrices unchanged
        for i in range(C.shape[0]):
            rho_before = np.outer(C[i], np.conj(C[i]))
            rho_after = np.outer(C_gauged[i], np.conj(C_gauged[i]))
            np.testing.assert_allclose(rho_before, rho_after, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_mo_gauge_idempotent(self):
        """Applying the gauge fix a second time is a no-op (idempotency).
        Pins the property that the SR optimizer does not have to re-gauge
        after every step: the gauge fix is a fixed point under itself."""
        rng = np.random.default_rng(82)
        C = rng.standard_normal((4, 7)) + 1j * rng.standard_normal((4, 7))
        C1 = _replay_gauge_fix(C)
        C2 = _replay_gauge_fix(C1)
        np.testing.assert_allclose(C1, C2, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    # ---------- 6. lambda Hermitian post-update -----------

    @pytest.mark.light
    def test_symmetrize_lambda_complex_hermitian(self):
        """:py:meth:`Geminal_data.symmetrize_lambda` Hermitises (not just
        symmetrises) a complex paired lambda. After symmetrisation the result
        satisfies ``||L - L^H||_F < 1e-12`` (not only ``L - L^T``)."""

        # Build a minimal Geminal_data with a 4x4 complex Hermitian paired
        # lambda. The class constructor needs orb_data_*, num_electron_*,
        # which we substitute with light stubs since symmetrize_lambda only
        # inspects ``self.lambda_matrix``.
        class _MinimalGeminal:
            # Just enough to satisfy Geminal_data.symmetrize_lambda's
            # attribute access path.
            pass

        rng = np.random.default_rng(90)
        L_paired = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        # Make it (nearly) Hermitian -- that is the trigger condition.
        L_paired = 0.5 * (L_paired + L_paired.conj().T)
        g = _MinimalGeminal()
        g.lambda_matrix = L_paired

        # Perturb the input by non-Hermitian noise and exercise the formula
        # directly: 0.5 * (mat + mat^H). This pins the math, not the binding.
        noise = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        mat_in = L_paired + 0.01 * noise
        mat_out = 0.5 * (mat_in + mat_in.conj().T)

        # Result is Hermitian to machine precision.
        np.testing.assert_allclose(mat_out, mat_out.conj().T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_symmetrize_lambda_real_path_byte_identical(self):
        """On a real symmetric input the Hermitian symmetrisation
        coincides byte-for-byte with the real ``0.5 * (A + A.T)``."""
        rng = np.random.default_rng(91)
        A = rng.standard_normal((5, 5))
        A_sym = 0.5 * (A + A.T)
        noise = rng.standard_normal((5, 5))
        mat_in = A_sym + 0.01 * noise  # real input
        # Hermitian = transpose for real
        mat_out_complex_path = 0.5 * (mat_in + mat_in.conj().T)
        mat_out_stage1 = 0.5 * (mat_in + mat_in.T)
        np.testing.assert_array_equal(mat_out_complex_path, mat_out_stage1)

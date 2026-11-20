"""Complex Bloch-pGTO AO leaves at a single twist.

Covers these deterministic complex periodic-orbital contracts:

1. Class hierarchy / MRO and the :func:`is_complex_aos` predicate.
2. **Gamma identity catch-net** at the AO level: a complex Bloch-pGTO leaf
   built at ``k = (0, 0, 0)`` reproduces the real-pgto pipeline walker-by-
   walker (``|Im| < atol``, ``Re ~= real`` to ``wf_eval`` tolerance) for value,
   grad, Laplacian, the fused value/grad/lap, and the overlap matrix.
3. **Non-Gamma Bloch-boundary identity**: AO satisfies
   ``chi_k(r + a) == e^{2 pi i k_frac . n} chi_k(r)`` for a single-cell shift.
4. **Image-phase invariants** of :func:`_promote_pgto_to_complex`:
   ``|phase| = 1`` and the array is aligned 1:1 with ``_image_indices``.
5. **Loader dispatch** of :func:`_apply_image_expansion_complex`: ``k_point=None``
   keeps the real leaf, otherwise the leaf gets promoted.

The MCMC ``|.|^2`` acceptance, complex ``e_L`` and ECP outer ``Re`` tests live
in their own files.
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

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_complex_dtype_np, get_dtype_np, get_tolerance
from jqmc._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_MEDIUM,
    RTOL_BIT_EXACT,
    RTOL_COMPLEX_PGTO_TRANSLATION_ENVELOPE,
    RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
)
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_cart_pgto_complex_data,
    AOs_cart_pgto_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_complex_data,
    AOs_sphe_pgto_data,
    _apply_image_expansion,
    _apply_image_expansion_complex,
    _promote_pgto_to_complex,
    build_AOs_cart_pgto_complex_data,
    build_AOs_sphe_pgto_complex_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
    compute_overlap_matrix,
    is_complex_aos,
)
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel.test_atomic_orbital import _make_aos_data

# Tolerances ---------------------------------------------------------------
# The Gamma identity is evaluated under pytest's per-test precision fixture.
# Do not cache its tolerance at module import: configuration has not yet been
# applied then, and a mixed run would silently retain the full-precision row.


def _make_pbc_fixture(ao_kind: str, k_point, *, nucleus_position=(5.0, 5.0, 5.0)):
    """Build a (structure_data, real_leaf, complex_leaf) triple at a chosen twist."""
    cell = CELL_FIXTURES["cubic_10bohr"]
    positions = np.array([nucleus_position], dtype=np.float64)
    # Real-path structure (k_point=None) for the real leaf.
    structure_real = make_pbc_structure(
        cell,
        positions=positions,
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )
    # Structure carries no k -- the twist lives on the AO leaf itself
    # (``AOs_GTO_base_data.k_point``). ``structure_cplx`` is an alias for
    # ``structure_real`` since the lattice / atom geometry is k-independent.
    structure_cplx = structure_real

    base = _make_aos_data(structure_real, ao_kind, l_max=2, uniform=True)
    base.sanity_check()
    real_leaf = _apply_image_expansion(base, structure_real, tol=AO_IMAGE_TRUNCATION_ATOL)

    # Build the complex leaf by promoting the same real-pgto leaf (so the
    # static fields are bit-equal except for ``k_point`` and ``_image_phase_factors``).
    cplx_leaf = _promote_pgto_to_complex(real_leaf, k_point)
    return structure_real, structure_cplx, real_leaf, cplx_leaf


def _walker_rng(ao_kind: str, seed: int = 20260615, n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed + (0 if ao_kind == "cart" else 1))
    return rng.uniform(0.5, 9.5, size=(n, 3)).astype(np.float64)


# ---------- 1. MRO / predicate -------------------------------------------


class TestComplexPGTO:
    @pytest.mark.light
    def test_mro_and_is_complex_aos_predicate(self):
        """Mixin sits FIRST in MRO so its overrides take precedence;
        :func:`is_complex_aos` only matches the complex leaves."""
        cart_mro = [c.__name__ for c in AOs_cart_pgto_complex_data.__mro__]
        sphe_mro = [c.__name__ for c in AOs_sphe_pgto_complex_data.__mro__]
        assert cart_mro.index("_ComplexBlochPGTOMixin") < cart_mro.index("AOs_cart_pgto_data"), (
            "mixin must precede AOs_cart_pgto_data in MRO so its _fold_image_axis / "
            "_compute_AOs* overrides resolve first; got MRO = " + str(cart_mro)
        )
        assert sphe_mro.index("_ComplexBlochPGTOMixin") < sphe_mro.index("AOs_sphe_pgto_data")

        # Predicate matches complex leaves and only them.
        real_cart = AOs_cart_pgto_data.__new__(AOs_cart_pgto_data)
        real_sphe = AOs_sphe_pgto_data.__new__(AOs_sphe_pgto_data)
        cplx_cart = AOs_cart_pgto_complex_data.__new__(AOs_cart_pgto_complex_data)
        cplx_sphe = AOs_sphe_pgto_complex_data.__new__(AOs_sphe_pgto_complex_data)
        assert not is_complex_aos(real_cart)
        assert not is_complex_aos(real_sphe)
        assert is_complex_aos(cplx_cart)
        assert is_complex_aos(cplx_sphe)

    # ---------- 2. Image-phase invariants ------------------------------------

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_promote_phase_unit_modulus_and_alignment(self, ao_kind: str):
        """Boundary-centred H in a cubic 10-bohr cell has unit-modulus image
        phases aligned with ``exp(2 pi i k_frac . n)`` at fixed k=(0.25,0,0).

        The nucleus at (0,5,5) produces nonzero image indices and nonreal
        phases for cart/sphe, exposing a conjugated phase without claiming
        coverage of all twists.
        """
        structure_real, structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(
            ao_kind, k_point=(0.25, 0.0, 0.0), nucleus_position=(0.0, 5.0, 5.0)
        )

        phase = np.asarray(cplx_leaf._image_phase_factors)
        # Per-shell ragged: one Bloch phase per kernel-AO (length M), aligned
        # 1:1 with the per-kernel ``_image_indices``.
        assert phase.shape == (int(cplx_leaf._num_ao_kernel),)
        assert phase.dtype == np.complex128
        # Unit modulus.
        np.testing.assert_allclose(
            np.abs(phase),
            np.ones(int(real_leaf._num_ao_kernel), dtype=np.float64),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_BIT_EXACT,
        )
        # 1:1 alignment with _image_indices. k lives on the AO leaf
        # (``AOs_GTO_base_data.k_point``), not on the structure.
        idx = np.asarray(cplx_leaf._image_indices, dtype=np.float64).reshape(-1, 3)
        assert np.any(idx != 0), "the fixture must realize nonzero image indices"
        assert np.any(np.abs(np.imag(phase)) > ATOL_PYTHON_FP64_MEDIUM), "the fixture must realize nonreal phases"
        k_frac = np.asarray(cplx_leaf.k_frac_np, dtype=np.float64)
        expected = np.exp(2j * np.pi * (idx @ k_frac)).astype(np.complex128)
        np.testing.assert_allclose(
            phase, expected, atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_BIT_EXACT, err_msg="Bloch image phase alignment"
        )

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_promote_phase_alignment_rejects_conjugated_phase(self, ao_kind: str, monkeypatch):
        """The fixed boundary-centred H fixture rejects conjugated image phases.

        Only the phase factors are corrupted; image indices and k=(0.25,0,0)
        remain unchanged. The same positive comparison must fail specifically
        on phase alignment, not unit modulus, for cart/sphe.
        """
        original_promote = _promote_pgto_to_complex

        def conjugated_promote(real_leaf, k_point):
            promoted = original_promote(real_leaf, k_point)
            return promoted.replace(_image_phase_factors=np.conj(promoted._image_phase_factors))

        monkeypatch.setattr(sys.modules[__name__], "_promote_pgto_to_complex", conjugated_promote)
        with pytest.raises(AssertionError, match="Bloch image phase alignment"):
            self.test_promote_phase_unit_modulus_and_alignment(ao_kind)

    @pytest.mark.light
    def test_apply_image_expansion_complex_dispatch(self):
        """``k_point=None`` -> real leaf as-is; ``k_point != None`` -> promoted complex leaf."""
        cell = CELL_FIXTURES["cubic_10bohr"]
        positions = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        s = make_pbc_structure(cell, positions=positions, atomic_numbers=(1,), element_symbols=("H",), atomic_labels=("H1",))

        base = _make_aos_data(s, "cart", l_max=1, uniform=True)
        out_real = _apply_image_expansion_complex(base, s, k_point=None, tol=AO_IMAGE_TRUNCATION_ATOL)
        assert isinstance(out_real, AOs_cart_pgto_data)
        assert not is_complex_aos(out_real)

        out_cplx = _apply_image_expansion_complex(base, s, k_point=(0.0, 0.0, 0.0), tol=AO_IMAGE_TRUNCATION_ATOL)
        assert isinstance(out_cplx, AOs_cart_pgto_complex_data)
        assert is_complex_aos(out_cplx)

    # ---------- 3. Gamma identity catch-net -----------------------------------

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_gamma_identity_value(self, ao_kind: str):
        """Complex AO at k = Gamma reproduces real AO walker-by-walker."""
        _structure_real, _structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.0, 0.0, 0.0))
        r = _walker_rng(ao_kind)

        real_val = np.asarray(compute_AOs(aos_data=real_leaf, r_carts=r))
        cplx_val = np.asarray(compute_AOs(aos_data=cplx_leaf, r_carts=r))
        assert real_val.dtype == get_dtype_np("ao_eval")
        assert cplx_val.dtype == get_complex_dtype_np("ao_bloch_eval")
        assert cplx_val.shape == real_val.shape

        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(
            np.imag(cplx_val), np.zeros((real_leaf.num_ao, r.shape[0]), dtype=np.float64), atol=atol, rtol=RTOL_BIT_EXACT
        )
        np.testing.assert_allclose(np.real(cplx_val), real_val, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_gamma_identity_grad(self, ao_kind: str):
        """Gamma identity grad."""
        _structure_real, _structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.0, 0.0, 0.0))
        r = _walker_rng(ao_kind)

        real_grad = tuple(np.asarray(g) for g in compute_AOs_grad(real_leaf, r))
        cplx_grad = tuple(np.asarray(g) for g in compute_AOs_grad(cplx_leaf, r))
        atol, rtol = get_tolerance("ao_grad_lap", "strict")
        for gr, gc in zip(real_grad, cplx_grad, strict=True):
            assert gc.dtype == get_complex_dtype_np("ao_grad_lap")
            np.testing.assert_allclose(
                np.imag(gc), np.zeros((real_leaf.num_ao, r.shape[0]), dtype=np.float64), atol=atol, rtol=RTOL_BIT_EXACT
            )
            np.testing.assert_allclose(np.real(gc), gr, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_gamma_identity_laplacian(self, ao_kind: str):
        """Gamma identity laplacian."""
        _structure_real, _structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.0, 0.0, 0.0))
        r = _walker_rng(ao_kind)

        real_lap = np.asarray(compute_AOs_laplacian(real_leaf, r))
        cplx_lap = np.asarray(compute_AOs_laplacian(cplx_leaf, r))
        assert cplx_lap.dtype == get_complex_dtype_np("ao_grad_lap")
        atol, rtol = get_tolerance("ao_grad_lap", "strict")
        np.testing.assert_allclose(
            np.imag(cplx_lap), np.zeros((real_leaf.num_ao, r.shape[0]), dtype=np.float64), atol=atol, rtol=RTOL_BIT_EXACT
        )
        np.testing.assert_allclose(np.real(cplx_lap), real_lap, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_gamma_identity_value_grad_lap_fused(self, ao_kind: str):
        """The fused entry point matches the real fused tuple element-by-element."""
        _structure_real, _structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.0, 0.0, 0.0))
        r = _walker_rng(ao_kind)

        real_tup = tuple(np.asarray(x) for x in compute_AOs_value_grad_lap(real_leaf, r))
        cplx_tup = tuple(np.asarray(x) for x in compute_AOs_value_grad_lap(cplx_leaf, r))
        assert len(real_tup) == len(cplx_tup) == 5
        for index, (x_real, x_cplx) in enumerate(zip(real_tup, cplx_tup, strict=True)):
            zone = "ao_eval" if index == 0 else "ao_grad_lap"
            atol, rtol = get_tolerance(zone, "strict")
            # The real value keeps ao_eval; the Bloch value retains its
            # cancellation-sensitive precision even at Gamma. Compare values
            # at the unchanged real-reference tolerance, and dtypes exactly.
            complex_zone = "ao_bloch_eval" if index == 0 else "ao_grad_lap"
            assert x_real.dtype == get_dtype_np(zone)
            assert x_cplx.dtype == get_complex_dtype_np(complex_zone)
            np.testing.assert_allclose(
                np.imag(x_cplx), np.zeros((real_leaf.num_ao, r.shape[0]), dtype=np.float64), atol=atol, rtol=RTOL_BIT_EXACT
            )
            np.testing.assert_allclose(np.real(x_cplx), x_real, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_gamma_identity_overlap_matrix(self, ao_kind: str):
        """Complex Bloch overlap at k=0 is Hermitian and reproduces the real Sym overlap."""
        _structure_real, _structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.0, 0.0, 0.0))
        S_real = np.asarray(compute_overlap_matrix(real_leaf))
        S_cplx = np.asarray(compute_overlap_matrix(cplx_leaf))
        assert S_cplx.dtype == np.complex128
        # Hermitian guarantee on the complex side (the override symmetrises with conj().T).
        np.testing.assert_allclose(S_cplx, S_cplx.conj().T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
        np.testing.assert_allclose(
            np.imag(S_cplx),
            np.zeros((real_leaf.num_ao, real_leaf.num_ao), dtype=np.float64),
            atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
            rtol=RTOL_BIT_EXACT,
        )
        # At Gamma the complex Bloch overlap reduces to the real symmetric overlap.
        np.testing.assert_allclose(
            np.real(S_cplx), S_real, atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, rtol=RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT
        )

    # ---------- 4. Non-Gamma Bloch-boundary identity --------------------------

    @pytest.mark.light
    def test_bloch_boundary_condition_non_gamma_cart(self):
        """``chi_k(r + a_alpha) == exp(2 pi i k_frac_alpha) * chi_k(r)``.

        Single-cell shift along each lattice vector; the wrap-restore phase
        + the truncated image sum together encode the Bloch boundary
        condition to image_truncation_atol.
        """
        k_frac = (0.25, -0.10, 0.40)
        structure_real, structure_cplx, _real_leaf, cplx_leaf = _make_pbc_fixture("cart", k_point=k_frac)
        cell = np.asarray(structure_real.cell, dtype=np.float64)

        rng = np.random.default_rng(20260615)
        # Wrap r firmly inside the home cell so the wrap-restore phase is exp(0)=1
        # (test exercises the +a side of the identity, not the wrap-restore alone).
        r = rng.uniform(2.0, 8.0, size=(6, 3)).astype(np.float64)

        chi_r = np.asarray(compute_AOs(aos_data=cplx_leaf, r_carts=r))
        # Tolerance loosens by image-truncation atol -- the two evaluations sum the
        # SAME finite image set, but the home cell is one step shifted on the +a
        # side, so the contributions near the truncation boundary differ by the
        # ``AO_IMAGE_TRUNCATION_ATOL`` envelope. Use a generous atol scaled by
        # max|chi| to absorb that.
        scale = float(np.max(np.abs(chi_r)))
        ao_atol, _ = get_tolerance("ao_eval", "strict")
        atol = max(scale * RTOL_COMPLEX_PGTO_TRANSLATION_ENVELOPE, ATOL_PYTHON_FP64_LOOSE, ao_atol)

        for alpha in range(3):
            a = cell[alpha]
            r_shift = r + a[None, :]
            chi_shift = np.asarray(compute_AOs(aos_data=cplx_leaf, r_carts=r_shift))
            phase = np.exp(2j * np.pi * k_frac[alpha])
            np.testing.assert_allclose(chi_shift, phase * chi_r, atol=atol, rtol=RTOL_BIT_EXACT)
        jax.clear_caches()

    # ---------- 5. Builders one-shot ----------------------------------------

    # ---------- 6. dtype-consistency sanity --------------------------------------

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_mos_sanity_check_rejects_dtype_mismatch(self, ao_kind: str):
        """``MOs_data.sanity_check`` rejects (real AO + complex MO) and
        (complex AO + real MO); accepts the matched pair."""
        from jqmc.molecular_orbital import MOs_data

        _structure_real, _structure_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.1, 0.2, 0.0))
        num_ao = real_leaf.num_ao
        C_real = np.random.RandomState(0).randn(num_ao, num_ao).astype(np.float64)
        C_cplx = (C_real + 1j * np.random.RandomState(1).randn(num_ao, num_ao)).astype(np.complex128)

        # Matched pairs pass.
        MOs_data(num_mo=num_ao, aos_data=real_leaf, mo_coefficients=C_real).sanity_check()
        MOs_data(num_mo=num_ao, aos_data=cplx_leaf, mo_coefficients=C_cplx).sanity_check()
        # Mismatched pairs raise.
        with pytest.raises(ValueError, match="complex Bloch-pGTO leaf requires complex128"):
            MOs_data(num_mo=num_ao, aos_data=cplx_leaf, mo_coefficients=C_real).sanity_check()
        with pytest.raises(ValueError, match="real AO leaf requires real"):
            MOs_data(num_mo=num_ao, aos_data=real_leaf, mo_coefficients=C_cplx).sanity_check()

    @pytest.mark.light
    def test_geminal_sanity_check_rejects_dtype_mismatch(self):
        """``Geminal_data.sanity_check`` enforces complex orb <=> complex lambda
        AND that both spin sectors agree on the twist regime."""
        from jqmc.determinant import Geminal_data
        from jqmc.molecular_orbital import MOs_data

        _s_real, _s_cplx, real_leaf, cplx_leaf = _make_pbc_fixture("cart", k_point=(0.0, 0.0, 0.0))
        num_ao = real_leaf.num_ao
        n = 2
        C_real = np.eye(num_ao, dtype=np.float64)
        C_cplx = (C_real + 0j).astype(np.complex128)
        mos_real = MOs_data(num_mo=num_ao, aos_data=real_leaf, mo_coefficients=C_real)
        mos_cplx = MOs_data(num_mo=num_ao, aos_data=cplx_leaf, mo_coefficients=C_cplx)
        lam_real = np.eye(num_ao, dtype=np.float64)
        lam_cplx = lam_real.astype(np.complex128)

        # Real <-> Real: passes.
        Geminal_data.build(
            num_electron_up=n,
            num_electron_dn=n,
            orb_data_up_spin=mos_real,
            orb_data_dn_spin=mos_real,
            lambda_matrix=lam_real,
        )
        # Complex <-> Complex: passes.
        Geminal_data.build(
            num_electron_up=n,
            num_electron_dn=n,
            orb_data_up_spin=mos_cplx,
            orb_data_dn_spin=mos_cplx,
            lambda_matrix=lam_cplx,
        )
        # Mixed orb regime: explicit rejection.
        with pytest.raises(ValueError, match="spin sectors disagree on twist regime"):
            Geminal_data.build(
                num_electron_up=n,
                num_electron_dn=n,
                orb_data_up_spin=mos_real,
                orb_data_dn_spin=mos_cplx,
                lambda_matrix=lam_real,
            )
        # Complex orb, real lambda: rejected by the orbital-leaf consistency rule.
        with pytest.raises(ValueError, match="complex orbital leaves require complex128"):
            Geminal_data.build(
                num_electron_up=n,
                num_electron_dn=n,
                orb_data_up_spin=mos_cplx,
                orb_data_dn_spin=mos_cplx,
                lambda_matrix=lam_real,
            )

    @pytest.mark.light
    def test_symmetrize_lambda_is_hermitian_when_complex(self):
        """``Geminal_data.symmetrize_lambda`` uses ``a.conj().T`` (Hermitian)
        when the stored lambda is complex; bit-identical on the real path."""
        from jqmc.determinant import Geminal_data
        from jqmc.molecular_orbital import MOs_data

        _s_real, _s_cplx, _real, cplx_leaf = _make_pbc_fixture("cart", k_point=(0.0, 0.0, 0.0))
        num_ao = cplx_leaf.num_ao
        n = 2
        C_cplx = np.eye(num_ao, dtype=np.complex128)
        mos = MOs_data(num_mo=num_ao, aos_data=cplx_leaf, mo_coefficients=C_cplx)
        # Construct a Hermitian lambda to trip the symmetry guard.
        rng = np.random.RandomState(42)
        A = (rng.randn(num_ao, num_ao) + 1j * rng.randn(num_ao, num_ao)).astype(np.complex128)
        lam_herm = 0.5 * (A + A.conj().T)
        g = Geminal_data.build(
            num_electron_up=n,
            num_electron_dn=n,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos,
            lambda_matrix=lam_herm,
        )
        # A non-Hermitian input that we want symmetrised; symmetrize_lambda
        # acts iff self.lambda_matrix is symmetric (which lam_herm is), so the
        # output is the Hermitian symmetrisation of the argument.
        mat = (rng.randn(num_ao, num_ao) + 1j * rng.randn(num_ao, num_ao)).astype(np.complex128)
        out = g.symmetrize_lambda(mat)
        # Hermitian, not transpose-symmetric.
        np.testing.assert_allclose(out, out.conj().T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
        expected = 0.5 * (mat + mat.conj().T)
        np.testing.assert_allclose(out, expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)

    @pytest.mark.light
    def test_AO_leaf_k_point_field_and_property(self):
        """``AOs_GTO_base_data.k_point`` field + ``k_frac_np`` property at the
        AO-leaf level (``k`` lives on the leaf itself; the structure carries no
        twist).
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        positions = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        s = make_pbc_structure(cell, positions=positions, atomic_numbers=(1,), element_symbols=("H",), atomic_labels=("H1",))
        base = _make_aos_data(s, "cart", l_max=1, uniform=True)
        real_leaf = _apply_image_expansion(base, s, tol=AO_IMAGE_TRUNCATION_ATOL)
        # Default real-pgto leaf: k_point=None -> zero k_frac.
        assert real_leaf.k_point is None
        np.testing.assert_array_equal(real_leaf.k_frac_np, np.zeros(3))
        # Promote with a twist.
        twist = (0.25, -0.1, 0.4)
        cplx_leaf = _promote_pgto_to_complex(real_leaf, twist)
        assert tuple(float(c) for c in cplx_leaf.k_point) == twist
        np.testing.assert_array_equal(cplx_leaf.k_frac_np, np.array(twist, dtype=np.float64))
        # ``with_negated_k()`` flips the sign.
        sibling = cplx_leaf.with_negated_k()
        assert tuple(float(c) for c in sibling.k_point) == tuple(-c for c in twist)

    # ---------- 8. ECP outer Re() wrap at Gamma is bit-identical ---------------

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
    def test_ecp_outer_re_wrap_gamma_bit_identical(self, ao_kind):
        """Real-Gamma ECP projection preserves every mesh coordinate, weight and total bit.

        Synthetic cubic-10-Bohr Cartesian/spherical pGTO, one up electron and
        no down electrons, SD/MO with only its first AO occupied, no Jastrow.
        The H centre is (5,5,5); the electron offset is (0.31,-0.27,0.49),
        without random sampling or conditioning selection. A nonzero s-channel
        ECP, NN=1, Nv=18 and identity rotation exercise both the forward and
        holomorphic public APIs in the same precision mode. Their real-Gamma
        outputs must have the same dtype and bytes. This is assembly-route
        parity, not an independent ECP physics reference. The real/complex
        leaf predicates remain separate checks, not substitutes for values.
        """
        import jax.numpy as jnp
        from jqmc.atomic_orbital import is_complex_aos
        from jqmc.coulomb_potential import (
            Coulomb_potential_data,
            compute_ecp_non_local_parts_nearest_neighbors,
            compute_ecp_non_local_parts_nearest_neighbors_holomorphic,
        )
        from jqmc.determinant import SD_data, compute_geminal_all_elements
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.molecular_orbital import MOs_data
        from jqmc.wavefunction import Wavefunction_data

        structure, _s_cplx, real_leaf, cplx_leaf = _make_pbc_fixture(ao_kind, k_point=(0.0, 0.0, 0.0))
        assert is_complex_aos(real_leaf) is False  # real path: outer Re wrap skipped
        assert is_complex_aos(cplx_leaf) is True  # complex representation, even at k=0
        coefficients = np.zeros((1, real_leaf.num_ao))
        coefficients[0, 0] = 1.0
        mos = MOs_data.build(aos_data=real_leaf, mo_coefficients=coefficients)
        geminal = SD_data.build(num_electron_up=1, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
        wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
        coulomb = Coulomb_potential_data(
            structure_data=structure,
            ecp_flag=True,
            z_cores=(0.0,),
            max_ang_mom_plus_1=(1,),
            num_ecps=2,
            nucleus_index=(0, 0),
            ang_moms=(0, 1),
            coefficients=(1.3, 1.0),
            exponents=(0.7, 0.4),
            powers=(2, 1),
        )
        r_up = jnp.asarray(structure.positions) + jnp.asarray([[0.31, -0.27, 0.49]])
        r_down = jnp.empty((0, 3))
        cond_g = np.linalg.cond(np.asarray(compute_geminal_all_elements(geminal, r_up, r_down)))
        assert np.isfinite(cond_g) and cond_g == 1.0
        args = (coulomb, wavefunction, r_up, r_down, jnp.eye(3))
        projected = compute_ecp_non_local_parts_nearest_neighbors(*args, NN=1, Nv=18)
        unprojected = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(*args, NN=1, Nv=18)
        assert np.any(np.asarray(projected[2]) != 0)
        for quantity, index in (("coordinates", 0), ("mesh-weights", 2), ("total", 3)):
            actual = np.asarray(projected[index]).reshape(-1)
            reference = np.asarray(unprojected[index]).reshape(-1)
            assert actual.size and actual.dtype == reference.dtype and np.all(np.isfinite(actual))
            actual_bytes, reference_bytes = actual.view(np.uint8), reference.view(np.uint8)
            np.testing.assert_array_equal(actual_bytes, reference_bytes, err_msg=quantity)
        jax.clear_caches()

    @pytest.mark.light
    def test_builder_one_shot_matches_promote(self):
        """``build_AOs_*_pgto_complex_data`` produces the same leaf as ``build`` + ``_promote``."""
        cell = CELL_FIXTURES["cubic_10bohr"]
        positions = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        s = make_pbc_structure(cell, positions=positions, atomic_numbers=(1,), element_symbols=("H",), atomic_labels=("H1",))
        k_point = (0.3, 0.1, 0.2)

        base = _make_aos_data(s, "cart", l_max=1, uniform=True)
        real_leaf = AOs_cart_pgto_data.build(
            structure_data=s,
            nucleus_index=base.nucleus_index,
            num_ao=base.num_ao,
            num_ao_prim=base.num_ao_prim,
            orbital_indices=base.orbital_indices,
            exponents=np.asarray(base.exponents, dtype=np.float64),
            coefficients=np.asarray(base.coefficients, dtype=np.float64),
            angular_momentums=base.angular_momentums,
            polynominal_order_x=base.polynominal_order_x,
            polynominal_order_y=base.polynominal_order_y,
            polynominal_order_z=base.polynominal_order_z,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        via_promote = _promote_pgto_to_complex(real_leaf, k_point)
        via_build = build_AOs_cart_pgto_complex_data(
            s,
            k_point=k_point,
            nucleus_index=base.nucleus_index,
            num_ao=base.num_ao,
            num_ao_prim=base.num_ao_prim,
            orbital_indices=base.orbital_indices,
            exponents=np.asarray(base.exponents, dtype=np.float64),
            coefficients=np.asarray(base.coefficients, dtype=np.float64),
            angular_momentums=base.angular_momentums,
            polynominal_order_x=base.polynominal_order_x,
            polynominal_order_y=base.polynominal_order_y,
            polynominal_order_z=base.polynominal_order_z,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        np.testing.assert_array_equal(np.asarray(via_promote._image_phase_factors), np.asarray(via_build._image_phase_factors))
        assert via_promote.num_ao == via_build.num_ao
        assert via_promote._n_image == via_build._n_image

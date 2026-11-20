"""Tests for the DTN polynomial Jastrow (J1 + J2 + J3 family).

Covers value, ratio, gradient and Laplacian agreement for the DTN polynomial:

  * Value: production ``compute_Jastrow_DTN`` vs the helper-free numpy
    oracle ``_compute_Jastrow_DTN_debug`` (molecular + periodic) x term
    presets x cutoff smoothness in {3, 4}.
  * Gradient / Laplacian: AD-based
    ``_compute_grads_and_laplacian_Jastrow_DTN_auto`` vs finite-difference
    of the numpy oracle (molecular + periodic).
  * Constraint / cusp correctness: e-e cusp, e-n cusp,
    f-term cuspless, permutation symmetry, cutoff-edge smoothness,
    constraint-matrix rank, cutoff safety guards.
  * Ratio consistency: single-electron move ratio via
    ``_compute_ratio_Jastrow_part_rank1_update`` /
    ``_compute_ratio_Jastrow_part_split_spin`` agrees with the
    brute-force ``exp(J(R') - J(R))`` reference (molecular + periodic).

Wavefunction-level checks:

  * Wavefunction-level fast-update coexistence:
    ``compute_coulomb_potential_fast`` (and its ECP twin) agree with the
    reference when a DTN Jastrow is present.
  * Force consistency: the autodiff position gradient of the
    DTN Jastrow value matches a finite-difference reference.
  * HDF5 round-trip: a ``Jastrow_DTN_data`` survives save/load via
    the generic ``Hamiltonian_data`` HDF5 pipeline.
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
import itertools
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel._gdd_fd_adapters import fd_case_parameters, record_fd_case

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_dtype_np, get_dtype_jnp, get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_DTN_COMOVING_CUTOFF_SCALE,
    ATOL_DTN_COMOVING_CUSP_TRANSPORT,
    ATOL_DTN_COMOVING_POLICY_FD,
    ATOL_DTN_COMOVING_POLICY_IDENTITY,
    ATOL_DTN_EN_CUSP_CONSTRAINT,
    ATOL_FD_VS_ANALYTIC_TIGHT,
    ATOL_INTERPOLATION_RELAXED,
    ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    ATOL_PYTHON_FP64_LOOSE,
    MAXULP_DTN_COMOVING_POLICY_IDENTITY_MIXED,
    RTOL_BIT_EXACT,
    RTOL_DTN_COMOVING_CUTOFF_SCALE,
    RTOL_DTN_COMOVING_POLICY_IDENTITY,
    RTOL_DTN_EN_CUSP_CONSTRAINT,
    RTOL_FD_VS_ANALYTIC_TIGHT,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import Geminal_data
from jqmc.derivatives.dtn import _md, dtn_cutoff_policy_fields
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    DTN_DEFAULT_CUTOFF_RATIO,
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _build_dtn_constraint_matrices,
    _dtn_gamma_constraint_matrix,
    _compute_grads_and_laplacian_Jastrow_DTN_auto,
    _dtn_default_cutoffs,
    _dtn_static_tuple,
    _dtn_cutoff_policy_full_coefficients,
    _DtnMinimumImage,
    _dtn_distance_rows,
    _dtn_minimum_image_plan,
    _dtn_f_atom_value_grad_hess,
    _dtn_f_atom_value_grad_hess_auto,
    _dtn_scalar_dist_with_derivs,
    _dtn_scalar_dist_with_derivs_auto,
    _compute_delta_J_DTN_mesh_dense,
    _compute_delta_J_DTN_per_config,
    compute_grads_and_laplacian_Jastrow_DTN,
    _compute_Jastrow_DTN_debug,
    _compute_ratio_Jastrow_part_debug,
    _compute_ratio_Jastrow_part_rank1_update,
    _compute_ratio_Jastrow_part_split_spin,
    _advance_j_dtn_candidate_mesh_state,
    _advance_jastrow_dtn_ratio_state,
    _advance_grads_laplacian_Jastrow_DTN_streaming_state,
    _compute_grads_and_laplacian_Jastrow_DTN_carry,
    _init_grads_laplacian_Jastrow_DTN_streaming_state,
    _init_j_dtn_candidate_mesh_state,
    _init_jastrow_dtn_ratio_state,
    compute_Jastrow_DTN,
    compute_ratio_Jastrow_one_move_mesh,
)
from jqmc.structure import (
    Structure_data,
    _minimum_image_displacement,
    _minimum_image_reduction_plan,
    _minimum_image_shell_shifts,
    _strain_structure_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests._ecp_core_electrons import core_electrons_of


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_dtn_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Check a homogeneous DTN derivative block using the central fidelity budget."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _check_dtn_mixed_fidelity(case_factory, analytical):
    """Compare fresh policy-specific analytical cases and reject a mixed-only fault."""
    mixed, full = mixed_vs_full(case_factory, analytical)
    blocks = mixed_fidelity_blocks(mixed, full)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in blocks:
        _assert_dtn_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
    mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_dtn_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)


def _assert_dtn_coincident_zero(values):
    """Keep the independent zero-extension convention in both precision modes."""
    value, gradient, laplacian = values
    atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
    np.testing.assert_allclose(float(value), 0.0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(gradient), np.zeros(3), atol=atol, rtol=rtol)
    np.testing.assert_allclose(float(laplacian), 0.0, atol=atol, rtol=rtol)


def _h2_structure() -> Structure_data:
    """Two H atoms 1.4 Bohr apart (molecular)."""
    return Structure_data.build(
        pbc_flag=False,
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )


def _h2_pbc_structure() -> Structure_data:
    """Two H atoms in the cubic 10 Bohr cell."""
    return make_pbc_structure(
        CELL_FIXTURES["cubic_10bohr"],
        positions=np.array([[2.0, 2.0, 2.0], [3.4, 2.0, 2.0]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )


def _term_flags(preset: str) -> tuple[bool, bool, bool]:
    """Translate a CLI preset string to (use_u, use_chi, use_f)."""
    if preset == "u":
        return True, False, False
    if preset == "u_chi":
        return True, True, False
    if preset == "u_f":
        return True, False, True
    if preset == "u_chi_f":
        return True, True, True
    raise ValueError(preset)


def _build_dtn(
    structure: Structure_data,
    *,
    preset: str = "u_chi_f",
    cutoff_smoothness: int = 4,
    seed: int = 0,
    L: float = 3.0,
    N_u: int = 4,
    N_chi: int = 4,
    N_eN: int = 2,
    N_ee: int = 2,
    random_init: bool = True,
) -> Jastrow_DTN_data:
    use_u, use_chi, use_f = _term_flags(preset)
    if structure.pbc_flag:
        R_ins = structure._wigner_seitz_inscribed_radius()
        L_f = min(L, 0.95 * 0.5 * R_ins)
    else:
        L_f = L
    # These are all-electron test structures.  State Z_eff explicitly: the
    # DTN factory cannot infer whether a bare atomic number has an ECP core.
    return Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=tuple(float(z) for z in np.unique(structure.atomic_numbers)),
        pseudized_per_species=tuple(False for _ in np.unique(structure.atomic_numbers)),
        use_u=use_u,
        use_chi=use_chi,
        use_f=use_f,
        cutoff_smoothness=cutoff_smoothness,
        n_order_u=N_u,
        n_order_chi=N_chi,
        n_order_eN=N_eN,
        n_order_ee=N_ee,
        L_u=None if structure.pbc_flag else L,
        L_chi=None if structure.pbc_flag else L,
        L_f=None if structure.pbc_flag else L_f,
        cutoff_ratio_u=L / R_ins if structure.pbc_flag else None,
        cutoff_ratio_chi=L / R_ins if structure.pbc_flag else None,
        cutoff_ratio_f=L_f / (0.5 * R_ins) if structure.pbc_flag else None,
        random_init=random_init,
        seed=seed,
    )


def _draw_electrons(structure: Structure_data, n_up: int, n_dn: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if structure.pbc_flag:
        cell = np.array([structure.vec_a, structure.vec_b, structure.vec_c], dtype=np.float64)
        # draw inside the home cell to avoid wrap-around noise in derivatives
        frac_up = rng.uniform(0.1, 0.9, size=(n_up, 3))
        frac_dn = rng.uniform(0.1, 0.9, size=(n_dn, 3))
        return frac_up @ cell, frac_dn @ cell
    return rng.normal(scale=0.5, size=(n_up, 3)), rng.normal(scale=0.5, size=(n_dn, 3))


# ---------------------------------------------------------------------------
# Value: production vs numpy oracle
# ---------------------------------------------------------------------------


def _basis_independent_coefficients(data: Jastrow_DTN_data, *, seed: int) -> Jastrow_DTN_data:
    """Replace the seeded independent coefficients by projections of seeded full-space draws.

    The constraint null-space bases ``M_*`` come from an SVD whose rotation
    inside the null space depends on the LAPACK build, so ``M @ indep`` for a
    fixed ``indep`` draw differs between environments.  Drawing ``g0`` in the
    full coefficient space and storing the least-squares ``indep`` with
    ``M indep ~= g0`` makes the full coefficients (``+ b``) the orthogonal
    projection of the seeded draw onto ``span(M)``, which is independent of
    the basis chosen inside that span and hence identical in every
    environment up to rounding.
    """
    rng = np.random.default_rng(seed)
    # The independent coefficients are stored in float64 in both modes (issue #781).
    dtype = np.float64
    replaced = {}
    for field, basis in (("alpha_indep", "M_alpha"), ("beta_indep", "M_beta"), ("gamma_indep", "M_gamma")):
        M = np.asarray(getattr(data, basis), dtype=np.float64)
        M = M.reshape(M.shape[0], -1, M.shape[-1])
        g0 = 0.01 * rng.standard_normal(M.shape[:2])
        indep = np.stack([np.linalg.lstsq(M[I], g0[I], rcond=None)[0] for I in range(M.shape[0])])
        replaced[field] = indep.astype(dtype)
    return data.replace(**replaced)


class TestDTNValue:
    """Value tests: production vs numpy oracle (mol + pbc)."""

    @pytest.mark.light
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
    @pytest.mark.parametrize("C", [3, 4])
    def test_Jastrow_dtn_part_molecular(self, preset, C):
        """Molecular values match the independent NumPy reference under same-precision strict bounds."""
        atol, rtol = get_tolerance("jastrow_eval", "strict")
        sd = _h2_structure()
        data = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=7)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=11)

        J_prod = float(compute_Jastrow_DTN(data, r_up, r_dn))
        J_orac = _compute_Jastrow_DTN_debug(data, r_up, r_dn)

        assert not np.isnan(J_prod) and not np.isnan(J_orac)
        np.testing.assert_allclose(J_prod, J_orac, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
    @pytest.mark.parametrize("C", [3, 4])
    @pytest.mark.parametrize("cutoff_scale", [1.0, 0.999, 1.001])
    def test_Jastrow_dtn_part_periodic(self, preset, C, cutoff_scale):
        """PBC values agree with independent NumPy inflation and transport at rest and both cutoff strains."""
        atol, rtol = get_tolerance_min(["jastrow_eval"], "strict")
        sd = _h2_pbc_structure()
        data = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=7)
        data = data.replace(cutoff_scale=cutoff_scale)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=11)

        J_prod = float(compute_Jastrow_DTN(data, r_up, r_dn))
        J_orac = _compute_Jastrow_DTN_debug(data, r_up, r_dn)

        assert not np.isnan(J_prod) and not np.isnan(J_orac)
        np.testing.assert_allclose(J_prod, J_orac, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("channel", ["u", "chi", "f"])
    @pytest.mark.parametrize("C", [3, 4])
    @pytest.mark.parametrize("cutoff_scale", [0.999, 1.001])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="fixed near-nuclear coordinates; seed 7 initializes coefficients only",
    )
    def test_periodic_transport_channel_matches_numpy(self, channel, C, cutoff_scale):
        """Each nonzero strained DTN channel matches NumPy at explicit near-nuclear electronic positions."""
        atol, rtol = get_tolerance_min(["jastrow_eval"], "strict")
        data = _basis_independent_coefficients(
            _build_dtn(_h2_pbc_structure(), preset="u_chi_f", cutoff_smoothness=C, seed=7), seed=7
        )
        data = data.replace(cutoff_scale=cutoff_scale, use_u=channel == "u", use_chi=channel == "chi", use_f=channel == "f")
        r_up = np.array([[1.87, 1.98, 2.31], [2.14, 1.64, 2.14]])
        r_dn = np.array([[1.88, 2.16, 1.93]])
        actual = float(compute_Jastrow_DTN(data, r_up, r_dn))
        reference = _compute_Jastrow_DTN_debug(data, r_up, r_dn)
        assert abs(reference) > 100 * atol, "channel must contribute a resolvable value"
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("channel", ["u", "chi", "f"])
    @pytest.mark.parametrize("C", [3, 4])
    @pytest.mark.parametrize("cutoff_scale", [0.999, 1.001])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="fixed near-nuclear coordinates; seed 7 initializes coefficients only",
    )
    def test_transport_verdict_rejects_missing_channel_transport(self, monkeypatch, channel, C, cutoff_scale):
        """Dropping one active channel's production transport breaks the same value verdict after a passing baseline."""
        import jqmc.jastrow_factor as jastrow

        self.test_periodic_transport_channel_matches_numpy(channel, C, cutoff_scale)
        name = {"u": "_dtn_inflate_alpha", "chi": "_dtn_inflate_beta", "f": "_dtn_inflate_gamma"}[channel]
        original = getattr(jastrow, name)

        def without_transport(data, dtype_jnp):
            return original(data.replace(cutoff_scale=1.0), dtype_jnp)

        monkeypatch.setattr(jastrow, name, without_transport)
        try:
            with pytest.raises(AssertionError, match="Not equal to tolerance"):
                self.test_periodic_transport_channel_matches_numpy(channel, C, cutoff_scale)
        finally:
            jax.clear_caches()


# ---------------------------------------------------------------------------
# Gradient / Laplacian: autodiff vs finite-difference of oracle
# ---------------------------------------------------------------------------


def _fd_grad_lap(data, r_up, r_dn, h_grad: float = 1e-5, h_lap: float = 5e-4):
    """Central finite difference of the oracle for grad and Laplacian.

    ``h_lap`` is 5e-4: the three-point second difference has an
    ``h^2/12 f''''`` truncation error, which at 1e-3 reaches 2.2e-7 on the
    C=3 u_f molecular fixture with its physical cusp biases (medium budget
    2.0e-7).  Halving the step cuts the truncation by
    four (5.5e-8) while the fp64 round-off ``~eps |J| / h^2`` stays ~1e-9.
    The comparison tolerances are unchanged.
    """

    def f(ru, rd):
        return _compute_Jastrow_DTN_debug(data, ru, rd)

    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    grad_up = np.zeros_like(r_up)
    grad_dn = np.zeros_like(r_dn)
    lap_up = np.zeros((n_up,), dtype=np.float64)
    lap_dn = np.zeros((n_dn,), dtype=np.float64)

    J0 = f(r_up, r_dn)
    for i in range(n_up):
        for d in range(3):
            rp = r_up.copy()
            rm = r_up.copy()
            rp[i, d] += h_grad
            rm[i, d] -= h_grad
            grad_up[i, d] = (f(rp, r_dn) - f(rm, r_dn)) / (2 * h_grad)
            rp2 = r_up.copy()
            rm2 = r_up.copy()
            rp2[i, d] += h_lap
            rm2[i, d] -= h_lap
            lap_up[i] += (f(rp2, r_dn) - 2 * J0 + f(rm2, r_dn)) / (h_lap**2)
    for i in range(n_dn):
        for d in range(3):
            rp = r_dn.copy()
            rm = r_dn.copy()
            rp[i, d] += h_grad
            rm[i, d] -= h_grad
            grad_dn[i, d] = (f(r_up, rp) - f(r_up, rm)) / (2 * h_grad)
            rp2 = r_dn.copy()
            rm2 = r_dn.copy()
            rp2[i, d] += h_lap
            rm2[i, d] -= h_lap
            lap_dn[i] += (f(r_up, rp2) - 2 * J0 + f(r_up, rm2)) / (h_lap**2)
    return grad_up, grad_dn, lap_up, lap_dn


# Analytic vs autodiff (_auto) cross-check. The analytic production path
# replaces the autodiff-via-jax.hessian reference for performance / memory --
# the two must agree to ~fp64 round-off everywhere the auto path is itself
# trusted (i.e. the auto-vs-FD tests above pass).
#
# Each (preset, C) pair is a separate pytest case so its presence is checked
# against the authored GDD requirements. Per-case cache clearing and the
# existing disable_jit region below preserve the DTN metadata-cache workaround.


def _check_analytic_and_auto_one(sd, preset, C):

    if is_mixed_precision_enabled():

        def case_factory():
            fresh_structure = sd.replace()
            data = _build_dtn(fresh_structure, preset=preset, cutoff_smoothness=C, seed=3)
            r_up, r_dn = _draw_electrons(fresh_structure, 2, 1, seed=4)
            return data, r_up, r_dn

        def analytical(case):
            data, r_up, r_dn = case
            expected_channels = tuple(name in preset.split("_") for name in ("u", "chi", "f"))
            assert (data.use_u, data.use_chi, data.use_f) == expected_channels
            assert data.cutoff_smoothness == C
            # Preserve the existing DTN static-metadata cache workaround.
            with jax.disable_jit():
                return compute_grads_and_laplacian_Jastrow_DTN(data, r_up, r_dn)

        _check_dtn_mixed_fidelity(case_factory, analytical)
        return

    data = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=3)
    expected_channels = tuple(name in preset.split("_") for name in ("u", "chi", "f"))
    assert (data.use_u, data.use_chi, data.use_f) == expected_channels, "DTN preset mismatch"
    assert data.cutoff_smoothness == C, "DTN cutoff smoothness mismatch"
    r_up, r_dn = _draw_electrons(sd, 2, 1, seed=4)

    # Disable JIT around both calls so the pytree cache (``_seen_qdds``) is
    # bypassed. Earlier tests in this module populate that cache with their
    # own Jastrow_DTN_data instances; ``Jastrow_DTN_data`` carries
    # ``pytree_node=False`` ``np.ndarray`` static fields (M_alpha / b_alpha
    # / M_beta / b_beta / M_gamma) which crash the cache-key equality check
    # whenever JAX tries to match a freshly-traced call against a cached
    # entry with structurally-similar-but-differently-valued metadata. This
    # is a pre-existing DTN-dataclass limitation, not specific to the
    # analytic implementation; disabling JIT for the comparison sidesteps
    # it without changing the data path under test.
    with jax.disable_jit():
        g_up_a, g_dn_a, l_up_a, l_dn_a = _compute_grads_and_laplacian_Jastrow_DTN_auto(data, r_up, r_dn)
        g_up_p, g_dn_p, l_up_p, l_dn_p = compute_grads_and_laplacian_Jastrow_DTN(data, r_up, r_dn)

    # Two exact implementations of the same grad/Laplacian (analytic vs
    # autodiff); the only disagreement is floating-point round-off in the
    # active ``jastrow_grad_lap`` zone, so scale the tolerance by its dtype
    # (fp64 in full mode, fp32 in mixed mode).
    atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
    np.testing.assert_allclose(np.asarray(g_up_p), np.asarray(g_up_a), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(g_dn_p), np.asarray(g_dn_a), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(l_up_p), np.asarray(l_up_a), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(l_dn_p), np.asarray(l_dn_a), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Kernel-level f-channel scalar kernel: analytic grad/Hessian agrees
# with the AD oracle on the 3-var tape.
#
# The whole-Jastrow analytic-vs-auto check above already covers the f
# channel indirectly, but a kernel-level test catches regressions in
# ``_dtn_f_atom_value_grad_hess`` (the inner product-rule + power-table
# kernel that drives the per-atom-pair contribution) before they fan out
# through the ``vmap(per_pair) -> vmap(per_atom)`` scaffold and become
# hard to localise. Mirrors the role the auto/analytic kernel oracles
# play for the one-body / two-body Jastrow families in
# ``tests/test_jqmc_kernel/test_jastrow.py``.
# ---------------------------------------------------------------------------


def _make_f_kernel_inputs(C, seed):
    """Build a representative ``(r_iI, r_jI, r_ij, gamma_I, L_I, N_eN, N_ee)``
    bundle for the f-channel scalar kernel.

    ``r_iI``, ``r_jI`` are kept strictly below ``L_I`` so the cutoff is
    non-zero (the on-edge / outside-cutoff branches are validated
    separately by ``test_dtn_cutoff_edge_smoothness_C4``). ``gamma_I`` is
    a small (N_eN+1, N_eN+1, N_ee+1) tensor of unconstrained random
    coefficients -- we are testing the kernel, not the DTN constraint
    matrices, so the symmetry / cusp structure is not required here.
    """
    rng = np.random.default_rng(seed)
    L_I = 3.0
    N_eN, N_ee = 2, 2
    r_iI = float(rng.uniform(0.3, L_I - 0.3))
    r_jI = float(rng.uniform(0.3, L_I - 0.3))
    r_ij = float(rng.uniform(0.3, 2.0 * L_I))
    gamma_I = rng.normal(size=(N_eN + 1, N_eN + 1, N_ee + 1))
    return r_iI, r_jI, r_ij, jnp.asarray(gamma_I, dtype=jnp.float64), L_I, C, N_eN, N_ee


# ---------------------------------------------------------------------------
# Kernel-level minimum-image distance: analytic grad/Laplacian
# agrees with the AD oracle on the 3-var displacement tape.
#
# ``_dtn_scalar_dist_with_derivs`` is the per-(i, I) / per-(i, j, I) atom
# distance helper that drives the u / chi / f channels of JDTN. On a
# 96-electron / 36-atom periodic cell the f channel alone fires it ~330k times
# per kinetic-energy eval, so the analytic PBC branch is where JDTN saves
# the most time.
# ---------------------------------------------------------------------------


def _cell_fixtures_for_dist():
    """Return ``(name, cell, inv_cell)`` triples for the PBC distance
    analytic-vs-auto sweep.

    Includes both an orthorhombic and a triclinic / oblique cell so the
    off-diagonal entries of ``inv_cell`` (which are zero for
    orthorhombic) are non-trivially exercised. Dimensions stay in the
    same ~10 Bohr ballpark as ``CELL_FIXTURES["cubic_10bohr"]`` used
    elsewhere in this file.
    """
    dtype = jnp.float64
    cell_cubic = jnp.eye(3, dtype=dtype) * 10.0
    cell_obl = jnp.asarray(
        np.array(
            [
                [10.0, 0.0, 0.0],
                [3.0, 9.5, 0.0],
                [1.5, 2.0, 11.0],
            ]
        ),
        dtype=dtype,
    )
    return (
        ("cubic", cell_cubic, jnp.linalg.inv(cell_cubic)),
        ("oblique", cell_obl, jnp.linalg.inv(cell_obl)),
    )


def _draw_near_nuclei(sd, n_up, n_dn, *, seed, spread=0.6):
    """Cluster electrons around random nuclei so the f-term is non-trivial."""
    atoms = np.asarray(sd._positions_cart_jnp)
    rng = np.random.default_rng(seed)
    base = atoms[rng.integers(0, len(atoms), size=n_up + n_dn)]
    e = base + rng.normal(0.0, spread, (n_up + n_dn, 3))
    return jnp.asarray(e[:n_up]), jnp.asarray(e[n_up:])


def _axial_shifts(alat: float, rot: np.ndarray | None = None) -> np.ndarray:
    """The 6-point LRDMC displacement lattice (optionally rotated)."""
    s = alat * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    return s if rot is None else s @ rot


def _mesh_combined_arrays(r_up: np.ndarray, r_dn: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference (m-major, k-minor, up-then-dn) mesh arrays -- the layout the
    fused kernel's row-major reshape assumes (gate 8 pins the production
    builder against the same layout)."""
    n_up, n_dn, n_k = r_up.shape[0], r_dn.shape[0], shifts.shape[0]
    new_up = np.repeat(r_up[None], (n_up + n_dn) * n_k, axis=0)
    new_dn = np.repeat(r_dn[None], (n_up + n_dn) * n_k, axis=0)
    for m in range(n_up):
        for k in range(n_k):
            new_up[m * n_k + k, m] += shifts[k]
    for m in range(n_dn):
        for k in range(n_k):
            new_dn[(n_up + m) * n_k + k, m] += shifts[k]
    return new_up, new_dn


class TestDTNFusedMesh:
    """Mesh-delta fusion gates.

    The fused (m, k)-lattice kernel `_compute_delta_J_DTN_mesh_dense` computes
    the WHOLE DTN mesh delta in one pass (u vs ALL electrons, chi vs ALL
    atoms, f j-side over ALL electrons for every atom). The exact-once
    combination happens inside `compute_ratio_Jastrow_one_move_mesh`,
    the single Jastrow-layer entry point for a one-move mesh: it excludes the
    generic DTN branch on EVERY return path of the rank-1 ratio and multiplies
    the fused factor itself, so no caller can double count.
    Gates: 1 parity vs the per-config oracle,
    2' oblique-cell crossing, 6 double-count POSITIVE CONTROL,
    7 spin channel + u-only pair, 8 flattening order.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
    def test_fused_mesh_delta_matches_per_config(self, pbc, preset):
        """Gate 1: fused (Ne, K) delta == the dense per-config oracle
        `_compute_delta_J_DTN_per_config`, every preset, molecular and PBC."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=23)
        r_up, r_dn = _draw_near_nuclei(sd, 3, 2, seed=29, spread=0.7)
        r_up, r_dn = np.asarray(r_up), np.asarray(r_dn)
        n_up, n_dn = r_up.shape[0], r_dn.shape[0]
        n_e = n_up + n_dn
        shifts = _axial_shifts(0.3)
        fused = np.asarray(_compute_delta_J_DTN_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(shifts)))
        assert fused.shape == (n_e, 6)
        for m in range(n_e):
            for k in range(6):
                mesh_label = f"{m}/{k}"
                new_up, new_dn = r_up.copy(), r_dn.copy()
                if m < n_up:
                    new_up[m] += shifts[k]
                else:
                    new_dn[m - n_up] += shifts[k]
                d_dense = _compute_delta_J_DTN_per_config(
                    dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(new_dn)
                )
                np.testing.assert_allclose(fused[m, k], float(d_dense), atol=atol, rtol=rtol)
        assert np.max(np.abs(fused)) > 100 * atol, "fixture must produce a non-trivial mesh delta"
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    @pytest.mark.parametrize(
        ("preset", "C"),
        [("u", 3), ("u_chi", 4), ("u_f", 3), ("u_chi_f", 4)],
    )
    def test_candidate_cache_multihop_matches_numpy_full_oracle(self, pbc, preset, C):
        """The projection-scoped DTN table remains a full-oracle twin.

        This is the cache-specific gate missing from the original tabpip-like
        implementation.  It checks the initial table and every table reached
        after a sequence of accepted up/down hops against the independent
        helper-free NumPy implementation of the complete DTN Jastrow.  The
        comparison therefore covers correction accumulation on unmoved rows,
        exact mover-row rebuilds, spin-channel selection, and u/chi/f cutoff
        crossings; it does not merely prove that the cached route was called.
        """
        # The optimized table stores its ratio in fp64, but its independent
        # full-recompute oracle evaluates the DTN value in the active
        # jastrow_eval zone.  The comparison is therefore bounded by the
        # weakest zone on this same-precision path.
        atol, rtol = get_tolerance_min(("jastrow_eval", "jastrow_ratio"), "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=43)
        r_up_jax, r_dn_jax = _draw_near_nuclei(sd, 3, 2, seed=47, spread=0.7)
        r_up = np.array(r_up_jax, copy=True)
        r_dn = np.array(r_dn_jax, copy=True)
        shifts = _axial_shifts(0.3)

        def candidates(up, dn):
            return (
                jnp.asarray(up[:, None, :] + shifts[None, :, :]),
                jnp.asarray(dn[:, None, :] + shifts[None, :, :]),
            )

        def full_oracle_table(up, dn):
            base = _compute_Jastrow_DTN_debug(dtn, up, dn)
            out = np.empty((up.shape[0] + dn.shape[0], shifts.shape[0]))
            for global_e in range(out.shape[0]):
                for k, shift in enumerate(shifts):
                    cand_up, cand_dn = up.copy(), dn.copy()
                    if global_e < up.shape[0]:
                        cand_up[global_e] += shift
                    else:
                        cand_dn[global_e - up.shape[0]] += shift
                    out[global_e, k] = _compute_Jastrow_DTN_debug(dtn, cand_up, cand_dn) - base
            return out

        candidate_up, candidate_dn = candidates(r_up, r_dn)
        state = _init_j_dtn_candidate_mesh_state(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), candidate_up, candidate_dn)

        checked_steps = []

        def assert_full_oracle(step_label):
            atol, rtol = get_tolerance_min(("jastrow_eval", "jastrow_ratio"), "strict")
            np.testing.assert_allclose(
                np.asarray(state.log_delta),
                full_oracle_table(r_up, r_dn),
                atol=atol,
                rtol=rtol,
            )
            checked_steps.append(step_label)

        assert_full_oracle("initial")
        # Both spins move; rows that have not moved for several hops exercise
        # accumulated four-term corrections rather than only fresh rebuilds.
        hops = [(True, 0, 0), (False, 1, 3), (True, 2, 4), (False, 0, 2), (True, 1, 5)]
        for hop_index, (is_up, electron, shift_index) in enumerate(hops):
            old_up, old_dn = r_up.copy(), r_dn.copy()
            if is_up:
                r_up[electron] += shifts[shift_index]
            else:
                r_dn[electron] += shifts[shift_index]
            candidate_up, candidate_dn = candidates(r_up, r_dn)
            state = _advance_j_dtn_candidate_mesh_state(
                dtn,
                state,
                jnp.asarray(is_up),
                jnp.asarray(electron),
                jnp.asarray(old_up),
                jnp.asarray(old_dn),
                jnp.asarray(r_up),
                jnp.asarray(r_dn),
                candidate_up,
                candidate_dn,
            )
            assert_full_oracle(f"hop-{hop_index}")
        assert checked_steps == ["initial", "hop-0", "hop-1", "hop-2", "hop-3", "hop-4"]
        jax.clear_caches()

    @pytest.mark.light
    def test_fused_oblique_cell_shift_crossing(self):
        """Gate 2': the counterexample cell
        ``[[10,0,0],[9,1,0],[0,0,10]]`` with a ROTATED axial shift that pulls
        the moved electron INTO an atom's cutoff its old position was outside
        of. Any distance-ball pre-selection around the old position fails
        here (round-based min-image is exact only for orthogonal cells); the
        consumed-tensor evaluation must match dense exactly."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        cell = np.array([[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
        sd = make_pbc_structure(
            cell,
            positions=np.array([[5.0, 0.5, 5.0]]),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        # R_inscribed of this cell is 0.5 (this cell's whole point is a
        # tiny WS ball vs a long lattice vector); all cutoffs must sit below.
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=7, L=0.45)
        L_f_max = float(np.max(np.asarray(dtn.L_f)))
        atom0 = np.asarray(sd._positions_cart_jnp)[0]
        theta = np.deg2rad(30.0)
        rot = np.array([[np.cos(theta), np.sin(theta), 0.0], [-np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
        # Geometry in L_f units (the default cutoffs clamp L_f well below the
        # requested L on this cell): candidate k=1 lands INSIDE the cutoff at
        # 0.5*L_f; the OLD position (candidate minus the rotated shift of
        # 0.8*L_f) sits outside EVERY image's L_f.
        shifts = _axial_shifts(0.8 * L_f_max, rot)
        inside_target = atom0 + np.array([0.5 * L_f_max, 0.0, 0.0])
        e_moved_old = inside_target - shifts[1]
        e_partner = atom0 + np.array([0.3 * L_f_max, 0.0, 0.1 * L_f_max])  # in-cutoff f/u partner
        r_up = np.stack([e_moved_old, e_partner])
        r_dn = atom0[None, :] + np.array([[0.0, 0.25 * L_f_max, -0.1 * L_f_max]])
        # Non-vacuity: verify the crossing geometry (min-image distances).
        d_old = float(_dtn_distance_rows(dtn, jnp.asarray(e_moved_old[None]), jnp.asarray(atom0[None]), jnp.float64)[0, 0])
        d_new = float(_dtn_distance_rows(dtn, jnp.asarray(inside_target[None]), jnp.asarray(atom0[None]), jnp.float64)[0, 0])
        assert d_old > L_f_max, f"old position must be OUTSIDE the cutoff (d={d_old}, L={L_f_max})"
        assert d_new < L_f_max, f"shifted position must be INSIDE the cutoff (d={d_new}, L={L_f_max})"
        fused = np.asarray(_compute_delta_J_DTN_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(shifts)))
        for m in range(3):
            for k in range(6):
                mesh_label = f"{m}/{k}"
                new_up, new_dn = r_up.copy(), r_dn.copy()
                if m < 2:
                    new_up[m] += shifts[k]
                else:
                    new_dn[m - 2] += shifts[k]
                d_dense = _compute_delta_J_DTN_per_config(
                    dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(new_dn)
                )
                np.testing.assert_allclose(fused[m, k], float(d_dense), atol=atol, rtol=rtol)
        # The crossing candidate itself must contribute (non-vacuous gate).
        assert abs(fused[0, 1]) > 100 * atol, "entering-shift delta must be significant"
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("spin_case", ["both", "dn_empty", "up_empty"])
    def test_fused_exact_once_double_count_positive_control(self, spin_case):
        """Gate 6: exact-once DTN wiring, asserted
        as a DIRECT POSITIVE CONTROL on every return path of the rank-1 ratio
        (both-spin AND the empty-spin full-recompute fallbacks):

        1. single-entry route: `compute_ratio_Jastrow_one_move_mesh` -- which
           owns BOTH the exclusion and the fused factor -- must equal the
           dense reference at (atol, rtol). It pins the entry point callers
           actually use, not a hand-assembled pair of half-contributions;
        2. DELIBERATE double count: the generic ratio (DTN branch enabled)
           times exp(delta_fused) -- i.e. exp(2*delta) by construction --
           must FAIL the SAME (atol, rtol) comparison on >= 1 entry. If it
           ever passes, this gate is RED: the sensitivity is asserted, not
           assumed (no saturation-magnitude precondition can do that)."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=11)
        jd = Jastrow_data(jastrow_dtn_data=dtn)
        if spin_case == "both":
            n_up, n_dn = 3, 2
        elif spin_case == "dn_empty":
            n_up, n_dn = 4, 0
        else:
            n_up, n_dn = 0, 4
        r_up_f, r_dn_f = _draw_near_nuclei(sd, n_up + n_dn, 0, seed=13, spread=0.5)
        all_e = np.asarray(r_up_f)
        r_up, r_dn = all_e[:n_up], all_e[n_up:]
        shifts = _axial_shifts(0.3)
        new_up, new_dn = _mesh_combined_arrays(r_up, r_dn, shifts)
        args = (jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(new_dn))
        # Dense reference: the generic branch owns DTN (flag False).
        ratio_dense = np.asarray(_compute_ratio_Jastrow_part_rank1_update(jd, *args))
        delta_fused = np.asarray(
            _compute_delta_J_DTN_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(shifts))
        )
        factor = np.exp(delta_fused.reshape(-1))
        # (1) single-entry route: the Jastrow layer must combine the excluded
        # generic branch and its fused factor into the whole ratio, exactly once.
        ratio_entry = np.asarray(compute_ratio_Jastrow_one_move_mesh(jd, *args, shifts=jnp.asarray(shifts)))
        np.testing.assert_allclose(ratio_entry, ratio_dense, atol=atol, rtol=rtol)
        # (2) POSITIVE CONTROL: both paths contributing == exp(2*delta) must
        # FAIL the same comparison (fixture has moderate |delta|, no
        # saturation -- verified by the assertion itself).
        double_counted = ratio_dense * factor
        assert np.any(~np.isclose(double_counted, ratio_dense, atol=atol, rtol=rtol)), (
            "double-count positive control PASSED the tolerance -- the gate has no sensitivity "
            "(fixture delta too small or saturated); the exact-once contract is unverifiable here"
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    def test_fused_spin_channel_asymmetric(self, pbc):
        """Gate 7: same/opposite-spin u assignment via GLOBAL index
        parity on an asymmetric (n_up != n_dn) fixture, molecular + PBC."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=19)
        r_up, r_dn = _draw_near_nuclei(sd, 4, 2, seed=31, spread=0.6)
        r_up, r_dn = np.asarray(r_up), np.asarray(r_dn)
        shifts = _axial_shifts(0.3)
        fused = np.asarray(_compute_delta_J_DTN_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(shifts)))
        for m in range(6):
            for k in range(6):
                mesh_label = f"{m}/{k}"
                new_up, new_dn = r_up.copy(), r_dn.copy()
                if m < 4:
                    new_up[m] += shifts[k]
                else:
                    new_dn[m - 4] += shifts[k]
                d_dense = _compute_delta_J_DTN_per_config(
                    dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(new_dn)
                )
                np.testing.assert_allclose(fused[m, k], float(d_dense), atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    def test_fused_u_only_pair_outside_f_lists(self):
        """Gate 7 companion: a pair within ``L_u`` but OUTSIDE every
        atom's ``L_f`` must contribute to u in the fused output. RED against
        any implementation that draws u partners from the f channel's active
        set (here NO electron is inside any atom's L_f, so that set is empty
        while the u pair is strongly coupled)."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_structure()
        dtn = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=[1.0],
            pseudized_per_species=[False],
            use_u=True,
            use_chi=True,
            use_f=True,
            cutoff_smoothness=4,
            n_order_u=4,
            n_order_chi=4,
            n_order_eN=2,
            n_order_ee=2,
            L_u=4.0,
            L_chi=0.8,
            L_f=0.8,
            random_init=True,
            seed=3,
        )
        atom0 = np.asarray(sd._positions_cart_jnp)[0]
        # Both electrons ~3 Bohr from every atom (>> L_f = L_chi = 0.8), yet
        # only 0.5 Bohr apart (<< L_u = 4): a u-ONLY pair.
        r_up = (atom0 + np.array([0.0, 3.0, 0.0]))[None, :]
        r_dn = (atom0 + np.array([0.5, 3.0, 0.0]))[None, :]
        # Fixture precondition: EVERY electron is outside EVERY atom's L_f.
        atoms = np.asarray(sd._positions_cart_jnp)
        all_e = np.concatenate([r_up, r_dn])
        r_eI = np.asarray(_dtn_distance_rows(dtn, jnp.asarray(all_e), jnp.asarray(atoms), jnp.float64))
        assert np.min(r_eI) > float(np.max(np.asarray(dtn.L_f))), "fixture must have an EMPTY f-active set"
        shifts = _axial_shifts(0.1)
        fused = np.asarray(_compute_delta_J_DTN_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(shifts)))
        for m in range(2):
            for k in range(6):
                mesh_label = f"{m}/{k}"
                new_up, new_dn = r_up.copy(), r_dn.copy()
                if m == 0:
                    new_up[0] += shifts[k]
                else:
                    new_dn[0] += shifts[k]
                d_dense = _compute_delta_J_DTN_per_config(
                    dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(new_dn)
                )
                np.testing.assert_allclose(fused[m, k], float(d_dense), atol=atol, rtol=rtol)
        assert np.max(np.abs(fused)) > 100 * atol, "the u-only pair must contribute to the fused delta"
        jax.clear_caches()

    @pytest.mark.light
    def test_fused_mesh_flattening_order_pinned(self):
        """Gate 8: pin the PRODUCTION mesh flattening (up electrons
        m-major / k-minor, then dn) against the actual ``r_*_carts_combined``
        arrays built by ``compute_discretized_kinetic_energy_fast_update``,
        and assert its e_L elements equal the INDEPENDENT slow reference
        ``compute_discretized_kinetic_energy`` (full per-config recompute of
        both determinant and Jastrow) on a REAL wavefunction (water ccECP)."""
        import os as _os

        from jqmc.determinant import compute_geminal_all_elements
        from jqmc.wavefunction import compute_discretized_kinetic_energy, compute_discretized_kinetic_energy_fast_update

        tre = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "trexio_example_files", "water_ccecp_ccpvqz.h5")
        if not _os.path.exists(tre):
            pytest.skip("water TREXIO fixture not present")
        sd, _aos, _mu, _md, geminal, _c = read_trexio_file(trexio_file=tre, store_tuple=True)
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=31, L=2.0)
        wf = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data(jastrow_dtn_data=dtn))
        nup, ndn = geminal.num_electron_up, geminal.num_electron_dn
        rng = np.random.default_rng(41)
        atoms = np.asarray(sd._positions_cart_jnp)
        base = atoms[rng.integers(0, len(atoms), size=nup + ndn)] + rng.normal(0, 0.7, (nup + ndn, 3))
        r_up, r_dn = jnp.asarray(base[:nup]), jnp.asarray(base[nup:])
        Ainv = jnp.linalg.pinv(compute_geminal_all_elements(geminal, r_up, r_dn))
        alat, RT = 0.3, jnp.eye(3)
        ru_c, rd_c, ek_fused = compute_discretized_kinetic_energy_fast_update(alat, wf, Ainv, r_up, r_dn, RT)
        ru_s, rd_s, ek_slow = compute_discretized_kinetic_energy(alat, wf, r_up, r_dn, RT)
        # Order pin: config g = m*6 + k (up block), then nup*6 + m*6 + k (dn).
        exp_up, exp_dn = _mesh_combined_arrays(np.asarray(r_up), np.asarray(r_dn), _axial_shifts(alat))
        np.testing.assert_array_equal(np.asarray(ru_c), exp_up)
        np.testing.assert_array_equal(np.asarray(rd_c), exp_dn)
        np.testing.assert_array_equal(np.asarray(ru_s), exp_up)
        np.testing.assert_array_equal(np.asarray(rd_s), exp_dn)
        # Fused fast-update route == the independent slow full-recompute route.
        atol, rtol = get_tolerance("wf_ratio", "medium")
        np.testing.assert_allclose(np.asarray(ek_fused), np.asarray(ek_slow), atol=atol, rtol=rtol)
        jax.clear_caches()


@contextmanager
def _fp64_parity_mode():
    """Run a MATH-parity gate at full precision regardless of the CLI mode.

    The fused drift/value kernel is contractually fp64 in BOTH modes
    (``_precision.py``: the derivatives package is fp64 by construction), but
    its dense oracles are forward-module kernels whose zones drop to fp32 in
    mixed mode -- on Ampere GPUs their einsums then run TF32 and the ORACLE
    noise (~1e-4) exceeds the mixed tolerance table. That noise says nothing
    about the kernel under test, so the parity comparison is pinned to full
    precision (public ``configure`` API, the ``test_mixed_precision`` pattern)."""
    from jqmc._precision import configure, is_mixed_precision_enabled

    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
        jax.clear_caches()
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")
            jax.clear_caches()


def _fp64_parity(fn):
    """Decorator form of :func:`_fp64_parity_mode` (keeps test bodies flat)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _fp64_parity_mode():
            return fn(*args, **kwargs)

    return wrapper


class TestDTNGeometryMeshWeighted:
    """One-electron DTN nuclear/strain deltas reduced with arbitrary mesh weights.

    The retained dense ``grad_dtn_wrt_{R,strain}`` calls are the independent
    oracle.  Production must obtain the same weighted result without rebuilding
    every untouched electron pair for every lattice candidate.
    """

    @staticmethod
    def _mesh(all_e, alat=0.3):
        shifts = alat * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        pts = np.concatenate([all_e[m][None, :] + shifts for m in range(all_e.shape[0])])
        eidx = np.repeat(np.arange(all_e.shape[0]), shifts.shape[0])
        return pts, eidx

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    @pytest.mark.parametrize("preset", ["u_chi_f", "u_chi", "u_f"])
    def test_weighted_geometry_matches_dense_oracle(self, pbc, preset):
        """Weighted one-move nuclear/strain deltas equal dense recomputation."""
        from jqmc.derivatives.dtn import (
            dtn_geometry_mesh_weighted,
            grad_dtn_wrt_R,
            grad_dtn_wrt_strain,
        )

        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=37)
        r_up, r_dn = _draw_near_nuclei(sd, 3, 2, seed=41, spread=0.7)
        r_up, r_dn = np.asarray(r_up), np.asarray(r_dn)
        all_e = np.concatenate([r_up, r_dn])
        pts, eidx = self._mesh(all_e)
        rng = np.random.default_rng(43)
        weights = rng.normal(size=len(pts)) + 1j * rng.normal(size=len(pts))

        got_R, got_eps = dtn_geometry_mesh_weighted(
            dtn,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
            jnp.asarray(pts),
            jnp.asarray(eidx),
            jnp.asarray(weights),
        )

        base_R = np.asarray(grad_dtn_wrt_R(dtn, jnp.asarray(r_up), jnp.asarray(r_dn)))
        base_eps = np.asarray(grad_dtn_wrt_strain(dtn, jnp.asarray(r_up), jnp.asarray(r_dn)))
        ref_R = np.zeros_like(base_R, dtype=np.complex128)
        ref_eps = np.zeros_like(base_eps, dtype=np.complex128)
        for pt, moved, weight in zip(pts, eidx, weights, strict=True):
            ru_c, rd_c = r_up.copy(), r_dn.copy()
            if moved < r_up.shape[0]:
                ru_c[moved] = pt
            else:
                rd_c[moved - r_up.shape[0]] = pt
            ref_R += weight * (np.asarray(grad_dtn_wrt_R(dtn, jnp.asarray(ru_c), jnp.asarray(rd_c))) - base_R)
            ref_eps += weight * (np.asarray(grad_dtn_wrt_strain(dtn, jnp.asarray(ru_c), jnp.asarray(rd_c))) - base_eps)

        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        np.testing.assert_allclose(np.asarray(got_R), ref_R, atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(got_eps), ref_eps, atol=atol, rtol=rtol)
        # Positive control: neither weighted channel may be vacuous.
        assert np.max(np.abs(ref_R)) > 100 * atol
        assert np.max(np.abs(ref_eps)) > 100 * atol
        jax.clear_caches()

    @pytest.mark.light
    def test_pbc_image_crossing_matches_dense_oracle(self):
        """A moved/partner f pair crossing the cell boundary keeps all virials
        on the same minimum-image vectors as the dense oracle."""
        from jqmc.derivatives.dtn import (
            dtn_geometry_mesh_weighted,
            grad_dtn_wrt_R,
            grad_dtn_wrt_strain,
        )

        sd = _h2_pbc_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=47)
        r_up = np.array([[9.9, 2.0, 2.0], [0.1, 2.0, 2.0]])
        r_dn = np.array([[2.5, 2.3, 2.1]])
        pts = np.array([[9.8, 2.0, 2.0], [0.2, 2.1, 2.0]])
        eidx = np.array([0, 1])
        weights = np.array([0.7 - 0.2j, -1.1 + 0.4j])
        got_R, got_eps = dtn_geometry_mesh_weighted(dtn, r_up, r_dn, jnp.asarray(pts), jnp.asarray(eidx), jnp.asarray(weights))
        base_R = np.asarray(grad_dtn_wrt_R(dtn, r_up, r_dn))
        base_eps = np.asarray(grad_dtn_wrt_strain(dtn, r_up, r_dn))
        ref_R = np.zeros_like(base_R, dtype=np.complex128)
        ref_eps = np.zeros_like(base_eps, dtype=np.complex128)
        for pt, moved, weight in zip(pts, eidx, weights, strict=True):
            ru_c = r_up.copy()
            ru_c[moved] = pt
            ref_R += weight * (np.asarray(grad_dtn_wrt_R(dtn, ru_c, r_dn)) - base_R)
            ref_eps += weight * (np.asarray(grad_dtn_wrt_strain(dtn, ru_c, r_dn)) - base_eps)
        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        np.testing.assert_allclose(np.asarray(got_R), ref_R, atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(got_eps), ref_eps, atol=atol, rtol=rtol)
        assert np.max(np.abs(ref_eps)) > 100 * atol


class TestDTNDriftMeshDense:
    """The fused rank-one DTN DRIFT-mesh kernel
    (`derivatives.dtn.dtn_drift_mesh_dense`) must equal the streamed dense
    oracle (`one_move_mesh._dtn_drift_probe` per config, and
    `_compute_delta_J_DTN_per_config` for the value lane) on every config of a
    kinetic-style one-move mesh. This kernel replaces the ~768-serial
    per-config drift stream that dominated the analytic-force execution."""

    @staticmethod
    def _mesh(all_e, alat=0.3):
        sh = alat * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        pts, eidx = [], []
        for m in range(all_e.shape[0]):
            for k in range(6):
                pts.append(all_e[m] + sh[k])
                eidx.append(m)
        return np.array(pts), np.array(eidx)

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    @pytest.mark.parametrize("preset", ["u_chi_f", "u_chi", "u_f"])
    def test_drift_mesh_matches_streamed_oracle(self, pbc, preset):
        """Every config's full drift field == the dense per-config oracle,
        incl. hops that leave/enter cutoffs (mesh built from +-alat shifts)."""
        from jqmc.derivatives.dtn import dtn_drift_mesh_dense
        from jqmc.derivatives.one_move_mesh import _dtn_drift_probe

        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=23)
        r_up, r_dn = _draw_near_nuclei(sd, 3, 2, seed=29, spread=0.7)
        r_up, r_dn = np.asarray(r_up), np.asarray(r_dn)
        all_e = np.concatenate([r_up, r_dn])
        n_up = r_up.shape[0]
        pts, eidx = self._mesh(all_e)
        dJ, O = dtn_drift_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(pts), jnp.asarray(eidx))
        dJ, O = np.asarray(dJ), np.asarray(O)
        atol_v, rtol_v = get_tolerance("jastrow_ratio", "strict")
        for c in range(pts.shape[0]):
            configuration_label = str(c)
            new_up, new_dn = r_up.copy(), r_dn.copy()
            m = eidx[c]
            if m < n_up:
                new_up[m] = pts[c]
            else:
                new_dn[m - n_up] = pts[c]
            ref = np.asarray(_dtn_drift_probe(dtn, jnp.asarray(new_up), jnp.asarray(new_dn)))
            np.testing.assert_allclose(O[c], ref, atol=atol, rtol=rtol)
            # value lane vs the established dense per-config delta oracle
            ref_v = float(
                _compute_delta_J_DTN_per_config(
                    dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(new_dn)
                )
            )
            np.testing.assert_allclose(dJ[c], ref_v, atol=atol_v, rtol=rtol_v)
        jax.clear_caches()

    @pytest.mark.light
    def test_drift_mesh_lattice_shift_invariance(self):
        """PROPERTY gate: shifting ANY electron by a full
        lattice vector must leave the kernel's ``(dJ, O)`` invariant
        (periodicity). Every min-image reduction satisfies this by
        construction, so the gate fails iff a RAW coordinate leaks into an
        assembled quantity -- a raw-coordinate leak, caught WITHOUT knowing
        the failure mode in advance. The PBC parity gate cannot see such a
        leak because its near-nuclei draws keep every
        raw e-e separation below the half cell (measured max 2.46 < 5.0
        incl. the mesh shifts), a regime where min_image == raw identically."""
        from jqmc.derivatives.dtn import dtn_drift_mesh_dense

        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        sd = _h2_pbc_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=23)
        r_up, r_dn = _draw_near_nuclei(sd, 3, 2, seed=29, spread=0.7)
        r_up, r_dn = np.asarray(r_up), np.asarray(r_dn)
        cellrow = np.asarray(sd.cell)[0]  # one full lattice vector
        outs = []
        for shift in (np.zeros(3), cellrow):
            ru = r_up.copy()
            ru[1] += shift  # translate one f-active electron by a lattice vector
            all_e = np.concatenate([ru, r_dn])
            pts, eidx = self._mesh(all_e)
            dJ, O = dtn_drift_mesh_dense(dtn, jnp.asarray(ru), jnp.asarray(r_dn), jnp.asarray(pts), jnp.asarray(eidx))
            outs.append((np.asarray(dJ), np.asarray(O)))
        np.testing.assert_allclose(outs[1][0], outs[0][0], atol=atol, rtol=rtol)
        np.testing.assert_allclose(outs[1][1], outs[0][1], atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    def test_drift_mesh_pbc_image_crossing(self):
        """Two f-active electrons straddling a
        cell boundary. The f e-e direction must keep the per-lane minimum-image
        vector; reconstructing it from raw coordinates (``src*sum(c/r) -
        sum((c/r)*e_j)``) drops the ``-n*cell`` term and gives a 3.4e-5
        drift error on exactly this geometry. The PBC parity gate never
        crosses an image in the e-e leg, so this topology is pinned here."""
        from jqmc.derivatives.dtn import dtn_drift_mesh_dense
        from jqmc.derivatives.one_move_mesh import _dtn_drift_probe

        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        sd = _h2_pbc_structure()  # atoms at x=2.0/3.4 in the cubic 10-bohr cell
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=23)
        # Both up electrons inside the f cutoff of the atom at (2,2,2) via the
        # minimum image, but on OPPOSITE sides of the x=0 boundary: the e-e
        # separation is 0.2 by minimum image and 9.8 by raw coordinates.
        r_up = np.array([[9.9, 2.0, 2.0], [0.1, 2.0, 2.0]])
        r_dn = np.array([[2.5, 2.3, 2.1]])
        pts = np.array([[9.8, 2.0, 2.0]])  # move electron 0 along -x
        eidx = np.array([0])
        dJ, O = dtn_drift_mesh_dense(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(pts), jnp.asarray(eidx))
        new_up = r_up.copy()
        new_up[0] = pts[0]
        ref = np.asarray(_dtn_drift_probe(dtn, jnp.asarray(new_up), jnp.asarray(r_dn)))
        # Positive control: the boundary-straddling pair must actually engage
        # the f e-e leg on the partner row (else this gate is vacuous).
        base = np.asarray(_dtn_drift_probe(dtn, jnp.asarray(r_up), jnp.asarray(r_dn)))
        assert np.max(np.abs(ref[1] - base[1])) > 100 * atol, "partner row must feel the image-crossing pair"
        np.testing.assert_allclose(np.asarray(O)[0], ref, atol=atol, rtol=rtol)
        # value lane (distance-only) sanity on the same config
        ref_v = float(
            _compute_delta_J_DTN_per_config(dtn, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new_up), jnp.asarray(r_dn))
        )
        atol_v, rtol_v = get_tolerance("jastrow_ratio", "strict")
        np.testing.assert_allclose(np.asarray(dJ)[0], ref_v, atol=atol_v, rtol=rtol_v)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("preset", ["u_chi_f", "u_chi"])
    def test_dtn_mesh_wiring_matches_streamed(self, preset):
        """one_move_mesh.dtn_mesh (the production entry) equals the streamed
        dense per-config oracle on the same mesh -- the fused kernel is
        the ONLY route, so this pins the wiring (index/flattening/base-field
        assembly), not a dispatch."""
        from jqmc.derivatives.one_move_mesh import _dtn_drift_probe, dtn_mesh

        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        atol_v, rtol_v = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_structure()
        dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=11)
        r_up, r_dn = _draw_near_nuclei(sd, 3, 2, seed=13, spread=0.6)
        r_up, r_dn = jnp.asarray(np.asarray(r_up)), jnp.asarray(np.asarray(r_dn))
        alat = 0.3
        sh = alat * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        pts_up = jnp.asarray(np.concatenate([np.asarray(r_up)[m] + sh for m in range(3)]))
        pts_dn = jnp.asarray(np.concatenate([np.asarray(r_dn)[m] + sh for m in range(2)]))
        eU = jnp.asarray(np.repeat(np.arange(3), 6), dtype=jnp.int32)
        jDl = jnp.asarray(np.repeat(np.arange(2), 6), dtype=jnp.int32)
        dJ, O = dtn_mesh(dtn, r_up, r_dn, pts_up, pts_dn, eU, jDl)
        r_up_np, r_dn_np = np.asarray(r_up), np.asarray(r_dn)
        n_up = r_up_np.shape[0]
        pts = np.concatenate([np.asarray(pts_up), np.asarray(pts_dn)])
        eidx = np.concatenate([np.asarray(eU), np.asarray(jDl) + n_up])
        for c in range(pts.shape[0]):
            configuration_label = str(c)
            new_up, new_dn = r_up_np.copy(), r_dn_np.copy()
            if eidx[c] < n_up:
                new_up[eidx[c]] = pts[c]
            else:
                new_dn[eidx[c] - n_up] = pts[c]
            ref = np.asarray(_dtn_drift_probe(dtn, jnp.asarray(new_up), jnp.asarray(new_dn)))
            np.testing.assert_allclose(np.asarray(O)[c], ref, atol=atol, rtol=rtol)
            ref_v = float(
                _compute_delta_J_DTN_per_config(
                    dtn, jnp.asarray(r_up_np), jnp.asarray(r_dn_np), jnp.asarray(new_up), jnp.asarray(new_dn)
                )
            )
            np.testing.assert_allclose(np.asarray(dJ)[c], ref_v, atol=atol_v, rtol=rtol_v)
        jax.clear_caches()

    @pytest.mark.light
    def test_streamed_fallback_is_loud(self, caplog):
        """A capability rejection keeps a production-reachable
        cell on the streamed dense path -- deliberate, but it must be LOUD (one
        warning per site+reason per process), naming the reason and the DTN
        dense-stream consequence."""
        import logging
        from types import SimpleNamespace

        from jqmc.derivatives import fast_mesh, one_move_mesh

        sd = _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=3)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=sd)
        # Bypass init validation exactly as the coverage-evidence suite does:
        # a non-pade/exp J2 is the simplest reachable rejection cell.
        jd = Jastrow_data(jastrow_two_body_data=j2, jastrow_dtn_data=dtn)
        jd = jd.replace(jastrow_two_body_data=jd.jastrow_two_body_data.replace(jastrow_2b_type="unknown"))
        # Model a regular, non-tiled determinant so the probe reaches the J2 gate.
        gd = SimpleNamespace(orb_data_up_spin=None)
        ham = SimpleNamespace(wavefunction_data=SimpleNamespace(geminal_data=gd, jastrow_data=jd))
        one_move_mesh._STREAMED_FALLBACK_WARNED.clear()
        with caplog.at_level(logging.WARNING, logger="jqmc"):
            assert one_move_mesh.fast_supported(ham) is False
            assert one_move_mesh.fast_supported(ham) is False  # once-guard: no second record
            assert fast_mesh.supported(ham) is False
        msgs = [r.getMessage() for r in caplog.records if "streamed full-recompute" in r.getMessage()]
        assert len(msgs) == 2, f"one warning per site expected, got {msgs}"
        assert any("ECP quadrature" in m for m in msgs) and any("S-prime" in m for m in msgs)
        assert all("non-pade/exp J2" in m for m in msgs)
        assert all("DENSE per-config drift stream" in m for m in msgs)
        one_move_mesh._STREAMED_FALLBACK_WARNED.clear()


class TestDTNGradLaplacian:
    """Grad/Laplacian tests (FD, analytic-vs-auto, kernel-level).

    Covers the autodiff-vs-finite-difference checks (``@skip_mixed_precision``
    on FD variants preserved -- FD round-off dominates under mixed precision),
    the analytic-vs-autodiff cross-checks at
    the whole-Jastrow level, and the kernel-level analytic-vs-auto
    oracles for the f-channel scalar kernel and the PBC minimum-image
    distance helper.
    """

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="PHY-JAS-CUSP-CONDITIONS-001",
    )
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
    @pytest.mark.parametrize("C", [3, 4])
    def test_numerical_and_auto_grads_and_laplacian_Jastrow_dtn_molecular(self, C, preset):
        """Molecular: AD grad / Laplacian matches central differences.

        Sweeps the same (preset x C) grid as the analytic-vs-auto check below
        so every active channel combination of the DTN family has a
        finite-difference anchor independent of the autodiff oracle (matches
        the unification done for the one-/two-/three-body Jastrow tests in
        ``tests/test_jqmc_kernel/test_jastrow.py``).
        """
        atol_g, rtol_g = get_tolerance("jastrow_grad_lap", "strict")
        atol_l, rtol_l = get_tolerance("jastrow_grad_lap", "medium")
        sd = _h2_structure()
        data = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=3)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=4)

        g_up_num, g_dn_num, l_up_num, l_dn_num = _fd_grad_lap(data, r_up, r_dn)
        g_up_ad, g_dn_ad, l_up_ad, l_dn_ad = _compute_grads_and_laplacian_Jastrow_DTN_auto(data, r_up, r_dn)

        np.testing.assert_allclose(np.asarray(g_up_ad), g_up_num, atol=atol_g, rtol=rtol_g)
        np.testing.assert_allclose(np.asarray(g_dn_ad), g_dn_num, atol=atol_g, rtol=rtol_g)
        np.testing.assert_allclose(np.asarray(l_up_ad), l_up_num, atol=atol_l, rtol=rtol_l)
        np.testing.assert_allclose(np.asarray(l_dn_ad), l_dn_num, atol=atol_l, rtol=rtol_l)
        record_fd_case(
            "fd-ad-dtn-grad-laplacian-molecular",
            pytest_node_id="tests/test_jqmc_kernel/test_jastrow_dtn.py::TestDTNGradLaplacian::test_numerical_and_auto_grads_and_laplacian_Jastrow_dtn_molecular",
            fixture_parameters=fd_case_parameters("fd-ad-dtn-grad-laplacian-molecular", C=C, preset=preset),
            coordinates=(r_up, r_dn),
            coordinate_generator="tests.test_jqmc_kernel.test_jastrow_dtn._draw_electrons",
            coordinate_seed=4,
            compared_pairs=((g_up_ad, g_up_num), (g_dn_ad, g_dn_num)),
            atol=atol_g,
            rtol=rtol_g,
            zones=("jastrow_grad_lap",),
            level="strict",
            stencil="central-first-order-2+central-second-order-2",
            step=1.0e-5,
            direction="cartesian-basis",
            additional_comparisons=(
                (((l_up_ad, l_up_num), (l_dn_ad, l_dn_num)), atol_l, rtol_l, ("jastrow_grad_lap",), "medium"),
            ),
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(reason="FD / numerical quadrature comparison dominated by float32 round-off")
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
    @pytest.mark.parametrize("C", [3, 4])
    def test_numerical_and_auto_grads_and_laplacian_Jastrow_dtn_periodic(self, C, preset):
        """PBC: AD grad / Laplacian matches central differences.

        Same (preset x C) coverage as the molecular sibling above for
        parity with the analytic-vs-auto sweep.
        """
        atol_g, rtol_g = get_tolerance("jastrow_grad_lap", "strict")
        atol_l, rtol_l = get_tolerance("jastrow_grad_lap", "medium")
        sd = _h2_pbc_structure()
        data = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=3)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=4)

        g_up_num, g_dn_num, l_up_num, l_dn_num = _fd_grad_lap(data, r_up, r_dn)
        g_up_ad, g_dn_ad, l_up_ad, l_dn_ad = _compute_grads_and_laplacian_Jastrow_DTN_auto(data, r_up, r_dn)

        np.testing.assert_allclose(np.asarray(g_up_ad), g_up_num, atol=atol_g, rtol=rtol_g)
        np.testing.assert_allclose(np.asarray(g_dn_ad), g_dn_num, atol=atol_g, rtol=rtol_g)
        np.testing.assert_allclose(np.asarray(l_up_ad), l_up_num, atol=atol_l, rtol=rtol_l)
        np.testing.assert_allclose(np.asarray(l_dn_ad), l_dn_num, atol=atol_l, rtol=rtol_l)
        record_fd_case(
            "fd-ad-dtn-grad-laplacian-periodic",
            pytest_node_id="tests/test_jqmc_kernel/test_jastrow_dtn.py::TestDTNGradLaplacian::test_numerical_and_auto_grads_and_laplacian_Jastrow_dtn_periodic",
            fixture_parameters=fd_case_parameters("fd-ad-dtn-grad-laplacian-periodic", C=C, preset=preset),
            coordinates=(r_up, r_dn),
            coordinate_generator="tests.test_jqmc_kernel.test_jastrow_dtn._draw_electrons",
            coordinate_seed=4,
            compared_pairs=((g_up_ad, g_up_num), (g_dn_ad, g_dn_num)),
            atol=atol_g,
            rtol=rtol_g,
            zones=("jastrow_grad_lap",),
            level="strict",
            stencil="central-first-order-2+central-second-order-2",
            step=1.0e-5,
            direction="cartesian-basis",
            additional_comparisons=(
                (((l_up_ad, l_up_num), (l_dn_ad, l_dn_num)), atol_l, rtol_l, ("jastrow_grad_lap",), "medium"),
            ),
        )
        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize(
        "preset,C", [("u", 3), ("u", 4), ("u_chi", 3), ("u_chi", 4), ("u_f", 3), ("u_f", 4), ("u_chi_f", 3), ("u_chi_f", 4)]
    )
    def test_analytic_and_auto_grads_and_laplacian_Jastrow_dtn_molecular(self, preset, C):
        """Production analytic ``compute_grads_and_laplacian_Jastrow_DTN`` must
        agree with the autodiff reference on the molecular path for every
        channel combination (preset) and both supported cutoff_smoothness
        values."""
        sd = _h2_structure()
        jax.clear_caches()
        _check_analytic_and_auto_one(sd, preset, C)
        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize(
        "preset,C", [("u", 3), ("u", 4), ("u_chi", 3), ("u_chi", 4), ("u_f", 3), ("u_f", 4), ("u_chi_f", 3), ("u_chi_f", 4)]
    )
    def test_analytic_and_auto_grads_and_laplacian_Jastrow_dtn_periodic(self, preset, C):
        """PBC counterpart of the analytic-vs-auto check."""
        sd = _h2_pbc_structure()
        jax.clear_caches()
        _check_analytic_and_auto_one(sd, preset, C)
        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize(
        "preset,C", [("u", 3), ("u", 4), ("u_chi", 3), ("u_chi", 4), ("u_f", 3), ("u_f", 4), ("u_chi_f", 3), ("u_chi_f", 4)]
    )
    def test_analytic_and_auto_grads_and_laplacian_Jastrow_dtn_li_ae(self, preset, C):
        """All-electron (Li atom, Z=3) analytic-vs-auto check.

        Complements the molecular H2 sweep by exercising a heavier nucleus
        (Z=3 vs Z=1), so the unconditional chi-channel constraint has a
        non-trivial cusp magnitude to honour. The H2 molecular test already
        exercises the all-electron path on Z=1; this extends coverage to Z>1.
        """
        import os

        from jqmc.trexio_wrapper import read_trexio_file

        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "Li_ae_ccpvdz_cart.h5")
        if not os.path.exists(trexio_path):
            pytest.skip("TREXIO fixture Li_ae_ccpvdz_cart.h5 not present.")
        sd, _aos, _mos_up, _mos_dn, _geminal, _coul = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )
        jax.clear_caches()
        _check_analytic_and_auto_one(sd, preset, C)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("C", [3, 4])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_analytic_and_auto_f_atom_value_grad_hess(self, C, seed):
        """Kernel-level analytic-vs-auto for the f-channel scalar kernel.

        ``_dtn_f_atom_value_grad_hess`` (analytic product-rule + power-table)
        must agree with ``_dtn_f_atom_value_grad_hess_auto`` (jax.grad +
        jax.hessian on the same scalar tape) to fp64 round-off for the value,
        the (3,) gradient and the (3, 3) Hessian, for both supported cutoff
        smoothness values and over a small seed sweep.
        """

        if is_mixed_precision_enabled():

            def case_factory():
                r_iI, r_jI, r_ij, gamma_I, L_I, C_v, N_eN, N_ee = _make_f_kernel_inputs(C, seed)
                dtype = get_dtype_jnp("jastrow_grad_lap")
                # Match the production DTN caller's zone; keep the source parameters deterministic.
                return (
                    jnp.asarray(r_iI, dtype=dtype),
                    jnp.asarray(r_jI, dtype=dtype),
                    jnp.asarray(r_ij, dtype=dtype),
                    jnp.asarray(gamma_I, dtype=dtype),
                    jnp.asarray(L_I, dtype=dtype),
                    C_v,
                    N_eN,
                    N_ee,
                    dtype,
                )

            def analytical(case):
                return _dtn_f_atom_value_grad_hess(*case)

            _check_dtn_mixed_fidelity(case_factory, analytical)
            jax.clear_caches()
            return

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
        r_iI, r_jI, r_ij, gamma_I, L_I, C_v, N_eN, N_ee = _make_f_kernel_inputs(C, seed)
        dtype_jnp = jnp.float64

        F_an, gF_an, hF_an = _dtn_f_atom_value_grad_hess(
            jnp.asarray(r_iI, dtype=dtype_jnp),
            jnp.asarray(r_jI, dtype=dtype_jnp),
            jnp.asarray(r_ij, dtype=dtype_jnp),
            gamma_I,
            L_I,
            C_v,
            N_eN,
            N_ee,
            dtype_jnp,
        )
        F_au, gF_au, hF_au = _dtn_f_atom_value_grad_hess_auto(
            jnp.asarray(r_iI, dtype=dtype_jnp),
            jnp.asarray(r_jI, dtype=dtype_jnp),
            jnp.asarray(r_ij, dtype=dtype_jnp),
            gamma_I,
            L_I,
            C_v,
            N_eN,
            N_ee,
            dtype_jnp,
        )

        np.testing.assert_allclose(np.asarray(F_an), np.asarray(F_au), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(gF_an), np.asarray(gF_au), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(hF_an), np.asarray(hF_au), atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name_idx", [0, 1])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_analytic_and_auto_scalar_dist_with_derivs_pbc(self, cell_name_idx, seed):
        """Kernel-level analytic-vs-auto for the PBC minimum-image distance.

        ``_dtn_scalar_dist_with_derivs`` (analytic Euclidean grad/Laplacian
        of the min-image displacement) must agree with
        ``_dtn_scalar_dist_with_derivs_auto`` (jax.grad + jax.jvp through the
        min-image distance; ``round`` has zero derivative a.e.) to fp64
        round-off for the value, the (3,) gradient and the scalar Laplacian,
        over both orthorhombic and triclinic cells and a small seed sweep of
        generic displacements inside the home cell.
        """

        if is_mixed_precision_enabled():

            def case_factory():
                _name, cell, inv_cell = _cell_fixtures_for_dist()[cell_name_idx]
                dtype = get_dtype_jnp("jastrow_grad_lap")
                rng = np.random.default_rng(seed)
                u = jnp.asarray(rng.uniform(-3.0, 3.0, size=3), dtype=dtype)
                r_b = jnp.zeros(3, dtype=dtype)
                basis_change, half_widths = _minimum_image_reduction_plan(np.asarray(cell))
                mic = _DtnMinimumImage(basis_change, _minimum_image_shell_shifts(half_widths))
                return u, r_b, True, cell, inv_cell, mic, dtype

            def analytical(case):
                return _dtn_scalar_dist_with_derivs(*case)

            _check_dtn_mixed_fidelity(case_factory, analytical)
            jax.clear_caches()
            return

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
        name, cell, inv_cell = _cell_fixtures_for_dist()[cell_name_idx]
        dtype = jnp.float64
        rng = np.random.default_rng(seed)
        # Stay away from the d^2 == 0 singularity (the helper returns 0 in
        # the limit; matching that is covered separately by
        # ``test_analytic_and_auto_scalar_dist_with_derivs_pbc_coincident``).
        u = jnp.asarray(rng.uniform(-3.0, 3.0, size=3), dtype=dtype)
        r_b = jnp.zeros(3, dtype=dtype)

        basis_change, half_widths = _minimum_image_reduction_plan(np.asarray(cell))
        mic = _DtnMinimumImage(basis_change, _minimum_image_shell_shifts(half_widths))
        r_an, gr_an, lr_an = _dtn_scalar_dist_with_derivs(u, r_b, True, cell, inv_cell, mic, dtype)
        r_au, gr_au, lr_au = _dtn_scalar_dist_with_derivs_auto(u, r_b, True, cell, inv_cell, mic, dtype)

        np.testing.assert_allclose(float(r_an), float(r_au), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(gr_an), np.asarray(gr_au), atol=atol, rtol=rtol)
        np.testing.assert_allclose(float(lr_an), float(lr_au), atol=atol, rtol=rtol)
        jax.clear_caches()
        del name

    @pytest.mark.light
    def test_analytic_and_auto_scalar_dist_with_derivs_pbc_coincident(self):
        """Both explicit-fp64 kernels obey the zero-extension convention at coincidence.

        The Euclidean distance is not differentiable at this point. This
        checks the implemented guards, not physical derivatives of the
        unextended distance. Values, gradients and Laplacians from both
        routes are checked; the analytic zero anchors remain independent
        of their mutual agreement.
        """

        def case_factory():
            _name, cell, inv_cell = _cell_fixtures_for_dist()[0]
            dtype = get_dtype_jnp("jastrow_grad_lap")
            u = jnp.zeros(3, dtype=dtype)
            basis_change, half_widths = _minimum_image_reduction_plan(np.asarray(cell))
            mic = _DtnMinimumImage(basis_change, _minimum_image_shell_shifts(half_widths))
            return u, u, True, cell, inv_cell, mic, dtype

        def analytical(case):
            values = _dtn_scalar_dist_with_derivs(*case)
            _assert_dtn_coincident_zero(values)
            return values

        if is_mixed_precision_enabled():
            _check_dtn_mixed_fidelity(case_factory, analytical)
            jax.clear_caches()
            return
        # The zero-extension mathematical anchor is independent of either AD or fidelity.
        analytical(case_factory())

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
        _, cell, inv_cell = _cell_fixtures_for_dist()[0]
        dtype = jnp.float64
        u = jnp.zeros(3, dtype=dtype)

        basis_change, half_widths = _minimum_image_reduction_plan(np.asarray(cell))
        mic = _DtnMinimumImage(basis_change, _minimum_image_shell_shifts(half_widths))
        r_an, gr_an, lr_an = _dtn_scalar_dist_with_derivs(u, u, True, cell, inv_cell, mic, dtype)
        r_au, gr_au, lr_au = _dtn_scalar_dist_with_derivs_auto(u, u, True, cell, inv_cell, mic, dtype)

        np.testing.assert_allclose(float(r_an), 0.0, atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(gr_an), np.zeros(3), atol=atol, rtol=rtol)
        np.testing.assert_allclose(float(lr_an), 0.0, atol=atol, rtol=rtol)
        np.testing.assert_allclose(float(r_au), 0.0, atol=atol, rtol=rtol)
        np.testing.assert_allclose(float(r_an), float(r_au), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(gr_an), np.asarray(gr_au), atol=atol, rtol=rtol)
        np.testing.assert_allclose(float(lr_an), float(lr_au), atol=atol, rtol=rtol)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# PBC minimum-image seam contract.  Random value/AD tests almost surely never
# land on the codimension-1 image boundary; these tests probe it deliberately.
# ---------------------------------------------------------------------------


class TestDTNMinImageSeams:
    """All DTN e--e fold paths agree in their supported domain and are C2-safe."""

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_ee_distance_paths_match_shared_mic_inside_supported_ball(self, cell_name):
        """Independent e--e implementations equal the shared exact MIC where used.

        DTN's fast fractional fold is not a general oblique-cell nearest-image
        solver. Its contract is narrower: every non-zero u/chi contribution has
        ``r < R_ins``, and every non-zero f contribution has ``r_ij < 2*L_f <
        R_ins``. Inside that ball the fractional fold and exact Wigner--Seitz
        MIC must be identical. This checks the value, analytic derivative,
        autodiff oracle, parameter-derivative helper and ratio-cache distance.
        """
        cell_np = CELL_FIXTURES[cell_name]
        sd = make_pbc_structure(cell_np)
        dtn = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            random_init=True,
            seed=17,
        )
        cell = jnp.asarray(cell_np, dtype=jnp.float64)
        inv_cell = jnp.linalg.inv(cell)
        radius_bound = sd._wigner_seitz_inscribed_radius()
        rng = np.random.default_rng(seed=20260806)
        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")

        for sample in range(12):
            sample_label = str(sample)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            base = direction * rng.uniform(0.1, 0.9) * radius_bound
            shift = rng.integers(-3, 4, size=3) @ cell_np
            a = jnp.asarray(base + shift, dtype=jnp.float64)
            b = jnp.zeros(3, dtype=jnp.float64)
            exact = np.asarray(_minimum_image_displacement(a - b, cell, inv_cell, sd._n_max))
            exact_r = float(np.linalg.norm(exact))

            mic = _dtn_minimum_image_plan(dtn)
            r_an, g_an, lap_an = _dtn_scalar_dist_with_derivs(a, b, True, cell, inv_cell, mic, jnp.float64)
            r_ad, g_ad, lap_ad = _dtn_scalar_dist_with_derivs_auto(a, b, True, cell, inv_cell, mic, jnp.float64)
            md = np.asarray(_md((a - b)[None, :], dtn)[0])
            rows = float(_dtn_distance_rows(dtn, a[None, :], b[None, :], jnp.float64)[0, 0])

            np.testing.assert_allclose(float(r_an), exact_r, atol=atol, rtol=rtol)
            np.testing.assert_allclose(float(r_ad), exact_r, atol=atol, rtol=rtol)
            np.testing.assert_allclose(rows, exact_r, atol=atol, rtol=rtol)
            np.testing.assert_allclose(md, exact, atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(g_an), exact / exact_r, atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(g_ad), exact / exact_r, atol=atol, rtol=rtol)
            np.testing.assert_allclose(float(lap_an), 2.0 / exact_r, atol=atol, rtol=rtol)
            np.testing.assert_allclose(float(lap_ad), 2.0 / exact_r, atol=atol, rtol=rtol)

    @staticmethod
    def _seam_observables(data, r_up, r_dn):
        value = np.asarray(compute_Jastrow_DTN(data, r_up, r_dn)).reshape(1)
        g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_DTN(data, r_up, r_dn)
        return np.concatenate(
            [
                value,
                np.asarray(g_up).reshape(-1),
                np.asarray(g_dn).reshape(-1),
                np.asarray(l_up).reshape(-1),
                np.asarray(l_dn).reshape(-1),
            ]
        )

    @pytest.mark.light
    @pytest.mark.parametrize("channel", ["u", "chi", "f"])
    def test_default_cutoff_channels_vanish_with_two_derivatives_at_fold_seam(self, channel):
        """Each hard-cutoff DTN channel is identically zero around its MIC seam.

        End-to-end check of the fold seam: test both
        one-sided configurations explicitly instead of relying on random walkers
        to hit a measure-zero boundary. Value, drift and Laplacian all vanish;
        an active in-support control ensures the selected channel is non-trivial.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        sd = make_pbc_structure(cell)
        data = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            use_u=channel == "u",
            use_chi=channel == "chi",
            use_f=channel == "f",
            n_order_u=3,
            n_order_chi=3,
            n_order_eN=3,
            n_order_ee=3,
            random_init=True,
            seed=23,
        )
        eps = 1.0e-6
        empty = np.zeros((0, 3), dtype=np.float64)
        if channel == "u":
            left = (np.array([[0.0, 0.0, 0.0], [5.0 - eps, 0.0, 0.0]]), empty)
            right = (np.array([[0.0, 0.0, 0.0], [5.0 + eps, 0.0, 0.0]]), empty)
            active = (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), empty)
        elif channel == "chi":
            left = (np.array([[5.0 - eps, 0.0, 0.0]]), empty)
            right = (np.array([[5.0 + eps, 0.0, 0.0]]), empty)
            active = (np.array([[1.0, 0.0, 0.0]]), empty)
        else:
            left = (np.array([[2.5 - 0.5 * eps, 0.0, 0.0], [-2.5 + 0.5 * eps, 0.0, 0.0]]), empty)
            right = (np.array([[2.5 + 0.5 * eps, 0.0, 0.0], [-2.5 - 0.5 * eps, 0.0, 0.0]]), empty)
            active = (np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]), empty)

        obs_left = self._seam_observables(data, *left)
        obs_right = self._seam_observables(data, *right)
        np.testing.assert_array_equal(obs_left, np.zeros_like(obs_left))
        np.testing.assert_array_equal(obs_right, np.zeros_like(obs_right))
        assert np.max(np.abs(self._seam_observables(data, *active))) > 1.0e-10

    @pytest.mark.light
    def test_unsafe_f_cutoff_negative_control_exposes_fold_gradient_jump(self):
        """Calibration: bypassing the PBC bound reproduces the forbidden f kink.

        Build the internally consistent L_f=3 polynomial as a molecular object,
        then replace only its structure with the cubic PBC twin to bypass the
        production sanity guard. The one-sided analytic drifts must show a clear
        jump, proving that the preceding safe-seam test is sensitive to the
        original bug rather than merely comparing two close floating values.

        The f coefficients are explicit, not a random_init draw: the raw e-e
        term ``gamma[0, 0, 2] = 1`` (``r_ij^2`` times the e-n cutoffs),
        projected onto the constraint null space by least squares in the
        stored basis. That projection depends only on the span, not on the
        basis inside it (a seeded draw gave a 3e-4 jump with one
        basis and passed with another), and it leaves a jump of about 0.12.
        """
        positions = np.zeros((1, 3), dtype=np.float64)
        pbc_sd = make_pbc_structure(
            CELL_FIXTURES["cubic_10bohr"],
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        mol_sd = Structure_data.build(
            pbc_flag=False,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        unsafe = Jastrow_DTN_data.build(
            structure_data=mol_sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            use_u=False,
            use_chi=False,
            use_f=True,
            n_order_eN=3,
            n_order_ee=3,
            L_f=3.0,
            random_init=False,
        )
        basis = np.asarray(unsafe.M_gamma, dtype=np.float64)[0]
        raw_f = np.zeros(basis.shape[:-1], dtype=np.float64)
        raw_f[0, 0, 2] = 1.0
        gamma_indep = np.linalg.lstsq(basis.reshape(-1, basis.shape[-1]), raw_f.ravel(), rcond=None)[0]
        unsafe = unsafe.replace(
            gamma_indep=gamma_indep[None, :].astype(np.asarray(unsafe.gamma_indep).dtype),
            structure_data=pbc_sd,
        )
        eps = 1.0e-6
        empty = np.zeros((0, 3), dtype=np.float64)
        left = np.array([[2.5 - 0.5 * eps, 0.0, 0.0], [-2.5 + 0.5 * eps, 0.0, 0.0]])
        right = np.array([[2.5 + 0.5 * eps, 0.0, 0.0], [-2.5 - 0.5 * eps, 0.0, 0.0]])
        grad_left = self._seam_observables(unsafe, left, empty)[1:7]
        grad_right = self._seam_observables(unsafe, right, empty)[1:7]
        assert np.max(np.abs(grad_left - grad_right)) > 1.0e-3


# ---------------------------------------------------------------------------
# Constraint / cusp correctness
# ---------------------------------------------------------------------------


def _one_sided_ee_deriv(data, base_pos: np.ndarray, spin_partner: str, h: float) -> float:
    """dJ_DTN / d r_12 |_{r12 = h/2} via forward difference (J(h) - J(h/2))/(h/2).

    Avoids the central-FD even-symmetry pitfall across the cusp at r_12 = 0:
    near coincidence J ~ c * |r_12|, which the central scheme would average
    away. The forward scheme on the positive side gives the right-derivative.
    """
    if spin_partner == "anti":
        r_up = base_pos.reshape(1, 3)
        # antiparallel: vary the dn electron position along +x
        r_dn_far = (base_pos + np.array([h, 0.0, 0.0])).reshape(1, 3)
        r_dn_near = (base_pos + np.array([0.5 * h, 0.0, 0.0])).reshape(1, 3)
        Jf = _compute_Jastrow_DTN_debug(data, r_up, r_dn_far)
        Jn = _compute_Jastrow_DTN_debug(data, r_up, r_dn_near)
    else:
        # parallel: both electrons same spin
        r_up_far = np.stack([base_pos, base_pos + np.array([h, 0.0, 0.0])], axis=0)
        r_up_near = np.stack([base_pos, base_pos + np.array([0.5 * h, 0.0, 0.0])], axis=0)
        r_dn = np.zeros((0, 3))
        Jf = _compute_Jastrow_DTN_debug(data, r_up_far, r_dn)
        Jn = _compute_Jastrow_DTN_debug(data, r_up_near, r_dn)
    return (Jf - Jn) / (0.5 * h)


class TestDTNConstraints:
    """Constraint / cusp correctness tests.

    Covers e-e cusp (anti/parallel), e-n cusp (AE), f-term cuspless at
    e-e and e-n, f-term permutation symmetry, cutoff-edge smoothness,
    constraint-matrix rank, and sanity rejections of low smoothness /
    over-large cutoffs in PBC.
    """

    @pytest.mark.light
    def test_legacy_cusp_off_data_is_rejected(self):
        """Legacy data with an unconstrained chi linear coefficient is unsafe."""
        sd = _h2_structure()
        legacy = _build_dtn(sd, L=3.0).replace(apply_en_cusp=False)
        with pytest.raises(ValueError, match=r"chi.*-chi'\(0\)/r.*357518.*VMC optimization"):
            legacy.sanity_check()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="PHY-JAS-CUSP-CONDITIONS-001",
    )
    def test_dtn_ee_cusp_antiparallel(self):
        """e-e cusp: dJ/dr_12 -> +1/2 from the positive side (anti-parallel)."""
        sd = _h2_structure()
        data = _build_dtn(sd, preset="u", seed=5, L=3.0)
        base = np.array([-5.0, 0.0, 0.0])  # far from nuclei
        # Take a sequence of shrinking h and extrapolate to h -> 0.
        derivs = [_one_sided_ee_deriv(data, base, "anti", h) for h in [1e-3, 1e-4, 1e-5]]
        np.testing.assert_allclose(derivs[-1], 0.5, atol=ATOL_INTERPOLATION_RELAXED, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="PHY-JAS-CUSP-CONDITIONS-001",
    )
    def test_dtn_ee_cusp_parallel(self):
        """e-e cusp: dJ/dr_12 -> +1/4 from the positive side (parallel)."""
        sd = _h2_structure()
        data = _build_dtn(sd, preset="u", seed=5, L=3.0)
        base = np.array([-5.0, 0.0, 0.0])
        derivs = [_one_sided_ee_deriv(data, base, "para", h) for h in [1e-3, 1e-4, 1e-5]]
        np.testing.assert_allclose(derivs[-1], 0.25, atol=ATOL_INTERPOLATION_RELAXED, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="PHY-JAS-CUSP-CONDITIONS-001",
    )
    def test_dtn_en_cusp_AE(self):
        """All-electron e-n cusp: dchi/dr|_0+ = -Z_eff."""
        sd = Structure_data.build(
            pbc_flag=False,
            positions=np.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        data = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            use_u=False,
            use_chi=True,
            use_f=False,
            cutoff_smoothness=4,
            n_order_chi=4,
            L_chi=3.0,
            L_u=3.0,
            L_f=3.0,
            random_init=False,
        )
        # With zero indep coefficients, beta_full is the cusp bias and chi'_I(0) = -Z.
        # Use a one-sided forward FD on the positive x-axis to avoid the cusp.
        r_dn = np.zeros((0, 3))

        def chi_at(r):
            return _compute_Jastrow_DTN_debug(data, np.array([[r, 0.0, 0.0]]), r_dn)

        # forward 2-point FD between r=h and r=h/2
        h = 1e-5
        deriv = (chi_at(h) - chi_at(0.5 * h)) / (0.5 * h)
        np.testing.assert_allclose(deriv, -1.0, atol=ATOL_INTERPOLATION_RELAXED, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_dtn_f_cuspless_at_ee(self):
        """f-term: ``Sum_{l+m=K} gamma_{I, l, m, 1} = 0`` for each ``K``.

        At electron coincidence ``r_i = r_j = r``, the e-e cuspless condition
        reduces to vanishing slabwise sums on the
        ``n = 1`` slab of ``gamma``. We verify this directly on the *inflated*
        coefficient tensor, with a random ``gamma_indep`` vector to exercise the
        full inflation map -- a FD-based check across the coincidence couples
        ``r_jI`` motion in with ``r_ij`` motion and is not a clean probe of the
        constraint.
        """
        sd = _h2_structure()
        data = _build_dtn(
            sd,
            preset="u_f",
            seed=11,
            L=3.0,
            N_eN=4,
            N_ee=4,
        )
        M_g = np.asarray(data.M_gamma)  # (n_species, N_eN+1, N_eN+1, N_ee+1, n_indep)
        rng = np.random.default_rng(0)
        gamma_indep = rng.standard_normal((M_g.shape[0], M_g.shape[-1]))
        gamma_full = np.einsum("Ilmni,Ii->Ilmn", M_g, gamma_indep)
        n_species, _, _, N_ee_p1 = gamma_full.shape[:4]
        N_eN = M_g.shape[1] - 1
        for I in range(n_species):
            for K in range(2 * N_eN + 1):
                slab_sum = 0.0
                for l in range(N_eN + 1):
                    m = K - l
                    if 0 <= m <= N_eN:
                        slab_sum += gamma_full[I, l, m, 1]
                assert abs(slab_sum) < ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, (
                    f"e-e cuspless slab sum K={K} (species {I}) = {slab_sum} != 0"
                )

    @pytest.mark.light
    def test_dtn_f_en_cuspless_at_en(self):
        """f-term grouped e-n condition at ``r_iI -> 0``, where ``r_ij=r_jI``."""
        sd = _h2_structure()
        data = _build_dtn(sd, preset="u_f", seed=12, N_eN=4, N_ee=4, L=3.0)
        M_g = np.asarray(data.M_gamma)
        C = data.cutoff_smoothness
        L = float(data.L_f[0])
        rng = np.random.default_rng(1)
        gamma_indep = rng.standard_normal((M_g.shape[0], M_g.shape[-1]))
        gamma_full = np.einsum("Ilmni,Ii->Ilmn", M_g, gamma_indep)
        N_eN = M_g.shape[1] - 1
        N_ee = M_g.shape[3] - 1
        for K in range(N_eN + N_ee + 1):
            grouped_sum = 0.0
            for m in range(N_eN + 1):
                n = K - m
                if 0 <= n <= N_ee:
                    grouped_sum += C * gamma_full[0, 0, m, n] - L * gamma_full[0, 1, m, n]
            assert abs(grouped_sum) < ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, (
                f"grouped e-n cuspless sum K={K}: {grouped_sum} != 0"
            )

    @pytest.mark.light
    def test_dtn_f_permutation_symmetry(self):
        """Production J_DTN is invariant under a same-spin electron permutation.

        The production evaluator and its reviewed debug reference must each be
        invariant, and must agree before and after the permutation.
        """
        sd = _h2_structure()
        data = _build_dtn(sd, preset="u_f", seed=13, L=3.0)
        r_up = np.array([[0.5, 0.1, 0.0], [-0.3, 0.4, 0.1]])
        r_dn = np.zeros((0, 3))
        J_orig = compute_Jastrow_DTN(data, r_up, r_dn)
        J_orig_debug = _compute_Jastrow_DTN_debug(data, r_up, r_dn)
        # swap the two electrons
        r_up_swap = r_up[::-1]
        J_swap = compute_Jastrow_DTN(data, r_up_swap, r_dn)
        J_swap_debug = _compute_Jastrow_DTN_debug(data, r_up_swap, r_dn)
        # Exact permutation symmetry: the two evaluations differ only by
        # round-off in the active ``jastrow_eval`` zone (fp64 full / fp32 mixed).
        atol, rtol = get_tolerance("jastrow_eval", "strict")
        np.testing.assert_allclose(J_orig, J_swap, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_orig, J_orig_debug, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_swap, J_swap_debug, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_dtn_f_null_basis_is_canonical(self):
        """The f-channel inflation basis is the canonical basis of its constraint null space.

        Inside the null space an SVD may return any rotation, and the rotation
        differs between LAPACK builds (aarch64 and x86_64 built
        different random_init DTN Jastrows from one seed). The reference here
        is built only from the null space itself: the projector
        ``I - pinv(C) C``, the projected unit vectors taken in index order
        whenever they raise the rank, and their QR factor with a positive R
        diagonal. That basis is unique for the span, so a production basis
        equal to it is the same on every platform.
        """
        from jqmc import jastrow_factor as jastrow_factor_module

        atol, rtol = get_tolerance("jastrow_eval", "strict")
        rng = np.random.default_rng(528)
        for n_order_eN, n_order_ee, smoothness, L_f in ((2, 2, 4, 2.5), (3, 2, 3, 1.7), (2, 3, 5, 3.1), (4, 4, 4, 4.75)):
            M_gamma = _build_dtn_constraint_matrices(
                n_species=1,
                z_eff_per_species=np.array([1.0]),
                pseudized_per_species=np.array([False]),
                use_u=False,
                use_chi=False,
                use_f=True,
                n_order_u=2,
                n_order_chi=2,
                n_order_eN=n_order_eN,
                n_order_ee=n_order_ee,
                L_u=5.0,
                L_chi=np.array([L_f]),
                L_f=np.array([L_f]),
                cutoff_smoothness=smoothness,
                normalize_basis=False,
            )[4]
            basis = np.asarray(M_gamma, dtype=np.float64)[0].reshape(-1, M_gamma.shape[-1])
            constraints = _dtn_gamma_constraint_matrix(n_order_eN, n_order_ee, smoothness, L_f)
            pseudoinverse = np.linalg.pinv(constraints)
            projector = np.eye(constraints.shape[1]) - pseudoinverse @ constraints
            # A relative cutoff on each growing prefix can count projector
            # roundoff as a new direction. Use one absolute floor at the full
            # projector scale, with the Frobenius norm product for pinv/matmul error.
            roundoff_scale = 1 + np.linalg.norm(pseudoinverse) * np.linalg.norm(constraints)
            rank_tol = projector.shape[0] * np.finfo(projector.dtype).eps * roundoff_scale
            pivots: list[int] = []
            for j in range(constraints.shape[1]):
                if np.linalg.matrix_rank(projector[:, pivots + [j]], tol=rank_tol) > len(pivots):
                    pivots.append(j)
            q, r = np.linalg.qr(projector[:, pivots])
            reference = q * np.sign(np.diag(r))
            assert basis.shape == reference.shape, (n_order_eN, n_order_ee, smoothness, L_f)
            np.testing.assert_allclose(basis, reference, atol=atol, rtol=rtol)
            # Same null space, different rows: an orthogonal row mixing changes
            # the rotation an SVD returns, but not the canonical basis.
            mixing = np.linalg.qr(rng.standard_normal((constraints.shape[0], constraints.shape[0])))[0]
            mixed_basis = jastrow_factor_module._canonical_null_space_basis(mixing @ constraints)
            np.testing.assert_allclose(mixed_basis, basis, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_dtn_cutoff_edge_smoothness_C4(self):
        """The sampled C=4 e-e Laplacian approaches the exact exterior zero.

        Use the u-only build so the test isolates the e-e cutoff edge; both
        electrons are placed far from any nucleus so the chi / f cutoffs and
        the autodiff through ``norm(r - R)`` are uninvolved.  Two interior
        distances establish decay towards the edge; two exterior distances
        establish the hard-cutoff zero rather than merely checking finiteness.
        """
        sd = _h2_structure()
        data = _build_dtn(sd, preset="u", cutoff_smoothness=4, seed=8, L=2.0)
        L = 2.0
        # Place the up electron far from both nuclei (H1 @ 0, H2 @ 1.4 x).
        r_up_base = np.array([[5.0, 5.0, 5.0]])
        laplacians = {}
        for shift in [-1e-3, -1e-5, 1e-5, 1e-3]:
            r_dn = r_up_base + np.array([[L + shift, 0.0, 0.0]])
            _, _, lap_up, _ = _compute_grads_and_laplacian_Jastrow_DTN_auto(data, r_up_base, r_dn)
            lap_val = float(lap_up[0])
            assert np.isfinite(lap_val), f"Laplacian non-finite at shift {shift}: {lap_val}"
            laplacians[shift] = lap_val

        assert abs(laplacians[-1e-5]) < abs(laplacians[-1e-3])
        np.testing.assert_array_equal(
            np.array([laplacians[1e-5], laplacians[1e-3]]),
            np.zeros(2),
        )
        jax.clear_caches()

    @pytest.mark.light
    def test_dtn_sanity_check_rejects_low_smoothness(self):
        """DTN sanity check rejects low smoothness."""
        sd = _h2_structure()
        with pytest.raises(ValueError, match="cutoff_smoothness"):
            Jastrow_DTN_data.build(
                structure_data=sd,
                z_eff_per_species=(1.0,),
                pseudized_per_species=(False,),
                cutoff_smoothness=2,
                L_u=3.0,
                L_chi=3.0,
                L_f=3.0,
            )

    @pytest.mark.light
    def test_dtn_sanity_check_rejects_large_cutoff_pbc(self):
        """PBC: L >= R_inscribed must raise.

        Periodic cutoffs are comoving ratios of ``R_inscribed`` (``L = ratio * R_ins``);
        absolute ``L_*`` are rejected outright, a ratio outside ``(0, 1)`` is rejected by
        ``build``, and a stored ``L_u >= R_inscribed`` is still caught by ``sanity_check``.
        """
        sd = _h2_pbc_structure()
        # R_inscribed = a/2 = 5.0 Bohr for the cubic_10bohr cell (true WS
        # inscribed radius; was the smooth-distance bound 10/pi ~ 3.18).
        with pytest.raises(ValueError, match="Absolute periodic DTN cutoffs are forbidden"):
            Jastrow_DTN_data.build(
                structure_data=sd,
                z_eff_per_species=(1.0,),
                pseudized_per_species=(False,),
                L_u=10.0,
                L_chi=10.0,
                L_f=10.0,
            )
        with pytest.raises(ValueError, match=r"cutoff_ratio_u must be in \(0,1\)"):
            Jastrow_DTN_data.build(
                structure_data=sd,
                z_eff_per_species=(1.0,),
                pseudized_per_species=(False,),
                cutoff_ratio_u=1.0,
            )
        dtn = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
        )
        with pytest.raises(ValueError, match="R_inscribed"):
            dtn.replace(L_u=10.0).sanity_check()


# ---------------------------------------------------------------------------
# Kato cusp slopes of the production evaluators for every smoothness C
# ---------------------------------------------------------------------------
#
# The evaluators use the cutoff factor ``(L - r)^C`` while the cusp
# constraints were once derived for ``(r - L)^C``.  The two differ by
# ``(-1)^C``, so every odd C imposed the opposite e-e and e-n slopes.  The
# checks above only used C=4 (or re-derived the slope from the same
# ``(r - L)^C`` algebra), and the odd-C tests only compared implementations of
# the same constrained coefficients.  These checks therefore measure the slope
# of the ACTUAL production gradient at each coalescence point, with no
# knowledge of the constraint algebra, for even and odd C.

#: Probe offset.  A power of two keeps every displaced coordinate exactly
#: representable in fp32 and fp64, so the probe direction is exact in both modes.
_CUSP_PROBE_STEP = 2.0**-12
#: Cutoff smoothness values.  C=2 is rejected by ``sanity_check`` (C >= 3).
_CUSP_PROBE_SMOOTHNESS = (3, 4, 5, 6)
#: (cutoff L, (N_u, N_chi, N_eN, N_ee)) shapes probed for each C.
_CUSP_PROBE_SHAPES = {
    "L1.5-low-order": (1.5, (2, 2, 2, 1)),
    "L2.5-high-order": (2.5, (6, 5, 3, 2)),
}
#: All-electron Li (Z_eff = 3) and pseudized Be (zero nuclear cusp), 4 Bohr apart.
_CUSP_PROBE_ATOMS = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
_CUSP_PROBE_ORIGIN_PBC = np.array([2.0, 2.0, 2.0])
#: Physical one-sided slopes: e-e antiparallel, e-e parallel, AE e-n (-Z_eff), ECP e-n.
_CUSP_PROBE_EXPECTED = np.array([0.5, 0.25, -3.0, 0.0])
_CUSP_PROBE_LABELS = ("ee-antiparallel", "ee-parallel", "en-all-electron", "en-pseudized")


def _cusp_probe_dtn(geometry: str, C: int, shape: str) -> Jastrow_DTN_data:
    """Random-coefficient u+chi+f DTN with one AE and one pseudized species."""
    L, (N_u, N_chi, N_eN, N_ee) = _CUSP_PROBE_SHAPES[shape]
    common = {
        "z_eff_per_species": (3.0, 2.0),
        "pseudized_per_species": (False, True),
        "use_u": True,
        "use_chi": True,
        "use_f": True,
        "cutoff_smoothness": C,
        "n_order_u": N_u,
        "n_order_chi": N_chi,
        "n_order_eN": N_eN,
        "n_order_ee": N_ee,
        "random_init": True,
        "seed": 104 + C,
    }
    labels = {"element_symbols": ("Li", "Be"), "atomic_labels": ("Li1", "Be1"), "atomic_numbers": (3, 4)}
    if geometry == "molecular":
        sd = Structure_data.build(pbc_flag=False, positions=_CUSP_PROBE_ATOMS, **labels)
        return Jastrow_DTN_data.build(structure_data=sd, L_u=L, L_chi=L, L_f=L, **common)
    sd = make_pbc_structure(CELL_FIXTURES["cubic_10bohr"], positions=_CUSP_PROBE_ATOMS + _CUSP_PROBE_ORIGIN_PBC, **labels)
    R_ins = sd._wigner_seitz_inscribed_radius()
    L_f = min(L, DTN_DEFAULT_CUTOFF_RATIO * 0.5 * R_ins)
    return Jastrow_DTN_data.build(
        structure_data=sd, cutoff_ratio_u=L / R_ins, cutoff_ratio_chi=L / R_ins, cutoff_ratio_f=L_f / (0.5 * R_ins), **common
    )


def _cusp_probe_slopes(data: Jastrow_DTN_data, grads_and_laplacian) -> np.ndarray:
    """One-sided coalescence slopes of the production DTN gradient.

    ``grads_and_laplacian(data, r_up, r_dn)`` is the evaluator under test
    (XLA or Rust).  One electron moves along ``+x`` through a coalescence
    point at signed offsets ``+-h``; ``D(h) = (g(h) - g(-h)) / 2`` of the
    x-gradient of the moving electron cancels every term that is smooth at
    the coalescence point (other electrons, other nuclei, the f channel's
    smooth part) and leaves ``slope + O(h)``.  Two Richardson steps over
    ``h, h/2, h/4`` remove the ``O(h)`` and ``O(h^2)`` terms.  Spectator
    electrons keep the u, chi and f channels all active at each probe.
    """
    atoms = np.asarray(data.structure_data.positions, dtype=np.float64)
    li, be = atoms[0], atoms[1]
    pair = li + np.array([0.0, 0.5, 0.0])  # e-e coalescence point inside every support
    spectator = li + np.array([0.25, -0.5, 0.375])
    x_hat = np.array([1.0, 0.0, 0.0])

    # (spin of the moving electron, its index, fixed up electrons, fixed down electrons, coalescence point)
    probes = (
        ("dn", 0, [pair, spectator], [None], pair),  # antiparallel pair
        ("up", 1, [pair, None], [spectator], pair),  # parallel pair
        ("up", 0, [None, spectator], [li + np.array([0.0, -0.75, 0.0])], li),  # all-electron nucleus
        ("up", 0, [None, spectator], [be + np.array([0.0, 1.0, 0.5])], be),  # pseudized nucleus
    )

    def radial_gradient(spin, index, ups, dns, origin, offset):
        moved = origin + offset * x_hat
        r_up = np.array([moved if r is None else r for r in ups], dtype=np.float64)
        r_dn = np.array([moved if r is None else r for r in dns], dtype=np.float64)
        g_up, g_dn, _l_up, _l_dn = grads_and_laplacian(data, r_up, r_dn)
        grad = np.asarray(g_up if spin == "up" else g_dn, dtype=np.float64)
        assert np.all(np.isfinite(grad)), "coalescence-probe gradient must be finite"
        return float(grad[index, 0])

    slopes = []
    for spin, index, ups, dns, origin in probes:
        half_jump = [
            0.5 * (radial_gradient(spin, index, ups, dns, origin, h) - radial_gradient(spin, index, ups, dns, origin, -h))
            for h in (_CUSP_PROBE_STEP, 0.5 * _CUSP_PROBE_STEP, 0.25 * _CUSP_PROBE_STEP)
        ]
        # Richardson weights cancelling O(h) and O(h^2): (D(h) - 6 D(h/2) + 8 D(h/4)) / 3.
        slopes.append((half_jump[0] - 6.0 * half_jump[1] + 8.0 * half_jump[2]) / 3.0)
    return np.asarray(slopes)


def _legacy_casino_sign_bias(data: Jastrow_DTN_data) -> Jastrow_DTN_data:
    """Legacy inhomogeneous biases: the ``(r - L)^C`` derivation multiplied them by ``(-1)^C``."""
    sign = (-1.0) ** int(data.cutoff_smoothness)
    return data.replace(
        b_alpha=_dtn_static_tuple(sign * np.asarray(data.b_alpha)),
        b_beta=_dtn_static_tuple(sign * np.asarray(data.b_beta)),
        policy_q_alpha=_dtn_static_tuple(sign * np.asarray(data.policy_q_alpha)),
        policy_q_beta=_dtn_static_tuple(sign * np.asarray(data.policy_q_beta)),
    )


class TestDTNCuspSlopeEverySmoothness:
    """Production XLA DTN gradients carry the Kato cusps for even AND odd C."""

    @pytest.mark.light
    @pytest.mark.parametrize("shape", sorted(_CUSP_PROBE_SHAPES))
    @pytest.mark.parametrize("C", _CUSP_PROBE_SMOOTHNESS)
    @pytest.mark.parametrize("geometry", ["molecular", "periodic"])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="explicit coalescence probes; the seed initializes DTN coefficients only",
    )
    def test_xla_gradient_has_kato_cusps_for_every_smoothness(self, geometry, C, shape):
        """The XLA analytic gradient gives e-e slopes 1/2 and 1/4, AE e-n slope -Z and ECP e-n slope 0.

        Judged at the active ``jastrow_grad_lap`` precision (fp64 full, fp32 mixed).
        """
        data = _cusp_probe_dtn(geometry, C, shape)
        slopes = _cusp_probe_slopes(data, compute_grads_and_laplacian_Jastrow_DTN)
        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        np.testing.assert_allclose(slopes, _CUSP_PROBE_EXPECTED, atol=atol, rtol=rtol, err_msg=str(_CUSP_PROBE_LABELS))
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("C", [3, 5])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="explicit coalescence probes; the seed initializes DTN coefficients only",
    )
    def test_xla_cusp_verdict_rejects_legacy_odd_smoothness_sign(self, monkeypatch, C):
        """The same verdict fails on the legacy odd-C biases after a passing baseline."""
        self.test_xla_gradient_has_kato_cusps_for_every_smoothness("molecular", C, "L1.5-low-order")
        module = sys.modules[__name__]
        original = module._cusp_probe_dtn
        monkeypatch.setattr(module, "_cusp_probe_dtn", lambda *args: _legacy_casino_sign_bias(original(*args)))
        try:
            with pytest.raises(AssertionError, match="Not equal to tolerance"):
                self.test_xla_gradient_has_kato_cusps_for_every_smoothness("molecular", C, "L1.5-low-order")
        finally:
            jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("C", [3, 5])
    @pytest.mark.parametrize("geometry", ["molecular", "periodic"])
    def test_serialized_pre_fix_odd_smoothness_biases_are_rejected(self, tmp_path, geometry, C):
        """An HDF5 DTN block written with legacy odd-C biases keeps them on load and is refused.

        The inflation biases are persisted metadata, so the corrected builder does
        not reach saved files.  The loader must therefore refuse them instead of
        evaluating the wrong e-e and e-n cusps; the current build of the same
        data round-trips and validates.
        """
        import h5py

        from jqmc.hamiltonians import _load_dataclass_from_hdf5, _save_dataclass_to_hdf5

        data = _cusp_probe_dtn(geometry, C, "L1.5-low-order")
        legacy = _legacy_casino_sign_bias(data)
        for name, payload in (("current", data), ("legacy", legacy)):
            with h5py.File(tmp_path / f"{name}.h5", "w") as f:
                _save_dataclass_to_hdf5(f.create_group("dtn"), payload)
        with h5py.File(tmp_path / "current.h5", "r") as f:
            _load_dataclass_from_hdf5(Jastrow_DTN_data, f["dtn"]).sanity_check()
        with h5py.File(tmp_path / "legacy.h5", "r") as f:
            loaded = _load_dataclass_from_hdf5(Jastrow_DTN_data, f["dtn"])
        np.testing.assert_array_equal(np.asarray(loaded.b_alpha), np.asarray(legacy.b_alpha))
        np.testing.assert_array_equal(np.asarray(loaded.b_beta), np.asarray(legacy.b_beta))
        with pytest.raises(
            ValueError, match=r"wrong sign for the \(L - r\)\^C cutoff form.*e-e cusp.*e-n cusp.*older jQMC version"
        ):
            loaded.sanity_check()


# ---------------------------------------------------------------------------
# Minimum-image bound + geometry-derived default cutoffs
# ---------------------------------------------------------------------------


class TestDTNCutoffPolicy:
    """WS inscribed radius and the geometry-derived default cutoffs."""

    @pytest.mark.light
    def test_ws_inscribed_radius_cubic(self):
        """Cubic a=10 -> true WS inscribed radius a/2 = 5.0 (was a/pi ~ 3.18)."""
        sd = _h2_pbc_structure()
        np.testing.assert_allclose(
            sd._wigner_seitz_inscribed_radius(), 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        # Molecular -> +inf so the bound check is a no-op.
        assert _h2_structure()._wigner_seitz_inscribed_radius() == float("inf")

    @pytest.mark.light
    def test_default_cutoffs_molecular(self):
        """H2 (R=1.4): L_u = min(R_max+10, cap) = 8.0; L_chi = L_f = 2.5*nn = 3.5."""
        sd = _h2_structure()
        L_u, L_chi, L_f = _dtn_default_cutoffs(sd, cutoff_fraction=0.95, cutoff_abs_cap=8.0)
        # R_max = 0.7 (centroid-to-atom) -> 0.7 + 10 = 10.7, capped at 8.0.
        np.testing.assert_allclose(L_u, 8.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        # nn = 1.4 -> 2.5 * 1.4 = 3.5 (below the 8.0 cap; no PBC bound).
        np.testing.assert_allclose(L_chi, 3.5, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(L_f, 3.5, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_default_cutoff_ratio_constant(self):
        """The single public default ratio is 0.95 for u, chi and f."""
        assert DTN_DEFAULT_CUTOFF_RATIO == 0.95

    @pytest.mark.light
    def test_default_cutoffs_pbc(self):
        """Periodic defaults are DTN_DEFAULT_CUTOFF_RATIO of each safe maximum."""
        sd = _h2_pbc_structure()
        rho = DTN_DEFAULT_CUTOFF_RATIO
        L_u, L_chi, L_f = _dtn_default_cutoffs(sd, cutoff_fraction=rho, cutoff_abs_cap=8.0)
        np.testing.assert_allclose(L_u, rho * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(L_chi, rho * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(L_f, rho * 2.5, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        # Defaults sit strictly inside the sanity_check bounds (L < R_ins,
        # 2*L_f < R_ins) -> the build() auto path passes sanity_check.
        assert L_u < 5.0 and L_chi < 5.0 and 2.0 * L_f < 5.0

    @pytest.mark.light
    def test_build_auto_cutoffs_applied(self):
        """build() with L=None wires the per-term defaults onto the dataclass.

        Cubic a=10 -> R_ins = 5.0, so the periodic defaults must resolve to
        ``0.95 * R_ins`` for u / chi and ``0.95 * R_ins / 2`` for f.
        """
        sd = _h2_pbc_structure()
        rho = DTN_DEFAULT_CUTOFF_RATIO
        data = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
        )  # all cutoffs auto
        np.testing.assert_allclose(
            float(np.asarray(data.L_u).reshape(())), 0.95 * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(np.asarray(data.L_chi), 0.95 * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            np.asarray(data.L_f), 0.95 * 0.5 * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(data.cutoff_ratio_u, rho, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            np.asarray(data.cutoff_ratio_chi), rho, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            np.asarray(data.cutoff_ratio_f), rho, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )

    @pytest.mark.light
    def test_build_explicit_cutoff_ratio_overrides_default(self):
        """An explicit ratio is honoured verbatim; the default never overrides it."""
        sd = _h2_pbc_structure()
        data = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            cutoff_ratio_u=0.75,
            cutoff_ratio_chi=0.75,
            cutoff_ratio_f=0.75,
        )
        np.testing.assert_allclose(
            float(np.asarray(data.L_u).reshape(())), 0.75 * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(np.asarray(data.L_chi), 0.75 * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            np.asarray(data.L_f), 0.75 * 0.5 * 5.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(data.cutoff_ratio_u, 0.75, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_periodic_absolute_cutoff_is_rejected(self):
        """Periodic DTN construction rejects an absolute cutoff instead of a comoving cutoff ratio."""
        sd = _h2_pbc_structure()
        with pytest.raises(ValueError, match="Absolute periodic DTN cutoffs are forbidden"):
            Jastrow_DTN_data.build(
                structure_data=sd,
                z_eff_per_species=(1.0,),
                pseudized_per_species=(False,),
                L_u=3.0,
            )

    @pytest.mark.light
    def test_full_coefficient_policy_tangent_matches_projected_manifold_fd(self):
        """The analytic transport is the derivative of closest-manifold projection."""
        data = _build_dtn(_h2_pbc_structure(), seed=41, N_u=4, N_chi=4, N_eN=2, N_ee=2)
        alpha = np.einsum("ski,si->sk", np.asarray(data.M_alpha), np.asarray(data.alpha_indep)) + np.asarray(data.b_alpha)
        beta = np.einsum("Iki,Ii->Ik", np.asarray(data.M_beta), np.asarray(data.beta_indep)) + np.asarray(data.b_beta)
        gamma = np.einsum("Ilmni,Ii->Ilmn", np.asarray(data.M_gamma), np.asarray(data.gamma_indep))
        analytical = tuple(np.asarray(x) for x in _dtn_cutoff_policy_full_coefficients(data, np.float64))
        h = 1.0e-5

        def project(sign):
            scale = np.exp(sign * h)
            Ma, ba, Mb, bb, Mg = _build_dtn_constraint_matrices(
                n_species=1,
                z_eff_per_species=np.asarray(data.z_eff_per_species),
                pseudized_per_species=np.asarray((False,)),
                use_u=True,
                use_chi=True,
                use_f=True,
                n_order_u=data.n_order_u,
                n_order_chi=data.n_order_chi,
                n_order_eN=data.n_order_eN,
                n_order_ee=data.n_order_ee,
                L_u=float(data.L_u) * scale,
                L_chi=np.asarray(data.L_chi) * scale,
                L_f=np.asarray(data.L_f) * scale,
                cutoff_smoothness=data.cutoff_smoothness,
                normalize_basis=False,
            )
            out_a = np.empty_like(alpha)
            for s in range(2):
                x = np.linalg.lstsq(Ma[s], alpha[s] - ba[s], rcond=1.0e-10)[0]
                out_a[s] = Ma[s] @ x + ba[s]
            x_b = np.linalg.lstsq(Mb[0], beta[0] - bb[0], rcond=1.0e-10)[0]
            out_b = (Mb[0] @ x_b + bb[0])[None, :]
            G = gamma.reshape(1, -1)
            M = Mg[0].reshape(G.shape[-1], -1)
            x_g = np.linalg.lstsq(M, G[0], rcond=1.0e-10)[0]
            out_g = (M @ x_g).reshape(gamma.shape)
            return out_a, out_b, out_g

        plus, minus = project(+1), project(-1)
        numerical = tuple((p - m) / (2.0 * h) for p, m in zip(plus, minus))
        for got, ref in zip(analytical, numerical):
            np.testing.assert_allclose(got, ref, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT)

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="Finite difference of policy grad/lap requires fp64.",
        alternative_guarantee_ref="MATH-DERIVAPI-CONSISTENCY-001",
    )
    def test_policy_value_grad_lap_matches_comoving_rebuild_fd(self):
        """Cutoff-policy derivatives of J, its gradients, and its Laplacian match a comoving-rebuild central FD."""
        sd = _h2_pbc_structure()
        data = _build_dtn(sd, seed=43, L=2.0, N_u=4, N_chi=4, N_eN=2, N_ee=2)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=44)
        alpha = np.einsum("ski,si->sk", np.asarray(data.M_alpha), np.asarray(data.alpha_indep)) + np.asarray(data.b_alpha)
        beta = np.einsum("Iki,Ii->Ik", np.asarray(data.M_beta), np.asarray(data.beta_indep)) + np.asarray(data.b_beta)
        gamma = np.einsum("Ilmni,Ii->Ilmn", np.asarray(data.M_gamma), np.asarray(data.gamma_indep))
        h = 1.0e-5

        def rebuilt(sign):
            scale = np.exp(sign * h)
            candidate = Jastrow_DTN_data.build(
                structure_data=sd,
                z_eff_per_species=data.z_eff_per_species,
                pseudized_per_species=(False,),
                use_u=True,
                use_chi=True,
                use_f=True,
                cutoff_smoothness=data.cutoff_smoothness,
                n_order_u=data.n_order_u,
                n_order_chi=data.n_order_chi,
                n_order_eN=data.n_order_eN,
                n_order_ee=data.n_order_ee,
                cutoff_ratio_u=data.cutoff_ratio_u * scale,
                cutoff_ratio_chi=np.asarray(data.cutoff_ratio_chi) * scale,
                cutoff_ratio_f=np.asarray(data.cutoff_ratio_f) * scale,
            )
            a_ind = np.stack(
                [
                    np.linalg.lstsq(
                        np.asarray(candidate.M_alpha)[s], alpha[s] - np.asarray(candidate.b_alpha)[s], rcond=1.0e-10
                    )[0]
                    for s in range(2)
                ]
            )
            b_ind = np.stack(
                [
                    np.linalg.lstsq(np.asarray(candidate.M_beta)[s], beta[s] - np.asarray(candidate.b_beta)[s], rcond=1.0e-10)[
                        0
                    ]
                    for s in range(beta.shape[0])
                ]
            )
            M = np.asarray(candidate.M_gamma).reshape(gamma.shape[0], -1, np.asarray(candidate.M_gamma).shape[-1])
            gflat = gamma.reshape(gamma.shape[0], -1)
            g_ind = np.stack([np.linalg.lstsq(M[s], gflat[s], rcond=1.0e-10)[0] for s in range(gamma.shape[0])])
            return candidate.replace(alpha_indep=a_ind, beta_indep=b_ind, gamma_indep=g_ind)

        plus, minus = rebuilt(+1), rebuilt(-1)
        Jp, Jm = compute_Jastrow_DTN(plus, r_up, r_dn), compute_Jastrow_DTN(minus, r_up, r_dn)
        gp = compute_grads_and_laplacian_Jastrow_DTN(plus, r_up, r_dn)
        gm = compute_grads_and_laplacian_Jastrow_DTN(minus, r_up, r_dn)
        numerical = ((Jp - Jm) / (2.0 * h),) + tuple((p - m) / (2.0 * h) for p, m in zip(gp, gm))
        analytical = dtn_cutoff_policy_fields(data, r_up, r_dn)
        for got, ref in zip(analytical, numerical):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(ref), atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT
            )

    @pytest.mark.light
    def test_full_strain_tensor_is_fixed_geometry_plus_isotropic_policy(self):
        """All nine strain components retain geometry; policy is diagonal."""
        from jqmc.derivatives.dtn import dtn_cutoff_policy_value, grad_dtn_wrt_strain
        from jqmc.structure import _strain_structure_data

        sd = _h2_pbc_structure()
        data = _build_dtn(sd, seed=47, L=2.0, N_u=4, N_chi=4, N_eN=2, N_ee=2)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=48)

        def fixed_cutoff_value(eps):
            F = jnp.eye(3, dtype=jnp.float64) + eps
            strained = data.replace(structure_data=_strain_structure_data(data.structure_data, F))
            return compute_Jastrow_DTN(strained, r_up @ F.T, r_dn @ F.T)

        fixed = jax.grad(fixed_cutoff_value)(jnp.zeros((3, 3), dtype=jnp.float64))
        policy = dtn_cutoff_policy_value(data, r_up, r_dn)
        reference = fixed + (policy / 3.0) * jnp.eye(3, dtype=jnp.float64)
        analytical = grad_dtn_wrt_strain(data, r_up, r_dn)
        np.testing.assert_allclose(
            np.asarray(analytical),
            np.asarray(reference),
            atol=ATOL_FD_VS_ANALYTIC_TIGHT,
            rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
        )

    @pytest.mark.light
    def test_sanity_check_rejects_seam_crossing_L_f(self):
        """A ratio reaching the f-channel seam must be rejected."""
        sd = _h2_pbc_structure()  # R_ins = 5.0
        with pytest.raises(ValueError, match=r"cutoff_ratio_f"):
            Jastrow_DTN_data.build(
                structure_data=sd,
                z_eff_per_species=(1.0,),
                pseudized_per_species=(False,),
                cutoff_ratio_f=2.6 / 2.5,
            )  # 2*2.6 = 5.2 > 5.0
        # L_chi at the same value is legal (bounded by R_ins, not R_ins/2).
        data = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),
            pseudized_per_species=(False,),
            cutoff_ratio_chi=2.6 / 5.0,
        )
        np.testing.assert_allclose(np.asarray(data.L_chi), 2.6, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)


# ---------------------------------------------------------------------------
# Ratio consistency (rank-1 / split-spin vs brute force)
# ---------------------------------------------------------------------------


def _move_one_electron(r_up: np.ndarray, r_dn: np.ndarray, spin: str, idx: int, axis: int, shift: float):
    new_up = r_up.copy()
    new_dn = r_dn.copy()
    if spin == "up":
        new_up[idx, axis] += shift
    else:
        new_dn[idx, axis] += shift
    return new_up, new_dn


@pytest.mark.light
@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
def test_dtn_ratio_streaming_state_matches_full_and_reinitialization(pbc, preset):
    """Cached DTN ratios and K accepted advances agree with full recomputation."""
    sd = _h2_pbc_structure() if pbc else _h2_structure()
    dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=17)
    jd = Jastrow_data(jastrow_dtn_data=dtn)
    r_up_np, r_dn_np = _draw_electrons(sd, 2, 1, seed=19)
    r_up = jnp.asarray(r_up_np)
    r_dn = jnp.asarray(r_dn_np)
    state = _init_jastrow_dtn_ratio_state(dtn, r_up, r_dn)
    atol, rtol = get_tolerance("jastrow_ratio", "strict")

    moves = (("up", 1, 0, 0.031), ("dn", 0, 2, -0.047), ("up", 0, 1, 0.023))
    for spin, idx, axis, shift in moves:
        new_up_np, new_dn_np = _move_one_electron(np.asarray(r_up), np.asarray(r_dn), spin, idx, axis, shift)
        new_up = jnp.asarray(new_up_np)
        new_dn = jnp.asarray(new_dn_np)
        ratio_cached = _compute_ratio_Jastrow_part_rank1_update(jd, r_up, r_dn, new_up[None], new_dn[None], dtn_state=state)
        ratio_full = jnp.exp(compute_Jastrow_DTN(dtn, new_up, new_dn) - compute_Jastrow_DTN(dtn, r_up, r_dn))
        np.testing.assert_allclose(np.asarray(ratio_cached[0]), np.asarray(ratio_full), atol=atol, rtol=rtol)

        global_idx = idx if spin == "up" else r_up.shape[0] + idx
        r_new = new_up[idx] if spin == "up" else new_dn[idx]
        all_e_new = jnp.concatenate((new_up, new_dn), axis=0)
        state = _advance_jastrow_dtn_ratio_state(dtn, state, global_idx, r_new, all_e_new)
        fresh = _init_jastrow_dtn_ratio_state(dtn, new_up, new_dn)
        for advanced_leaf, fresh_leaf in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(fresh)):
            np.testing.assert_allclose(np.asarray(advanced_leaf), np.asarray(fresh_leaf), atol=atol, rtol=rtol)
        r_up, r_dn = new_up, new_dn

    # Rejection is represented by selecting the old carry; it must be exact.
    rejected = jax.lax.cond(jnp.bool_(False), lambda s: fresh, lambda s: s, state)
    for rejected_leaf, state_leaf in zip(jax.tree_util.tree_leaves(rejected), jax.tree_util.tree_leaves(state)):
        np.testing.assert_array_equal(np.asarray(rejected_leaf), np.asarray(state_leaf))


# ---------------------------------------------------------------------------
# Rank-1 fast-update sweep -- mirrors the J3 ``_RATIO_RANK1_PARAM_SETS``
# pattern in tests/test_jqmc_kernel/test_jastrow.py (production-emulator path), adapted to
# DTN's preset / cutoff_smoothness axes. Verifies that the DTN delta-ratio
# fast update (``_compute_delta_J_DTN_per_config*``) agrees with the
# brute-force ``_compute_ratio_Jastrow_part_debug`` reference across:
#   * channel selection ``preset`` (u / u_chi / u_f / u_chi_f)
#   * cutoff smoothness ``C`` (3 / 4)
#   * move ``pattern`` (all_moved / none_moved / mixed)
#   * grid count ``n_grid`` (1 = MCMC single-move, 3/6 = ECP nonlocal mesh)
#   * molecular vs PBC distance metric
#
# No ``use_dtn_state`` toggle: DTN has no AO matmul to amortize, so unlike
# J3 there is no streaming-cache code path to switch on/off (see the note in
# ``_compute_delta_J_DTN_per_config``).
# ---------------------------------------------------------------------------


_RATIO_RANK1_DTN_PARAM_SETS = [
    # (preset, C, pattern, n_grid) -- representative covering set, as for J3:
    # the default DTN combo (``u_chi_f``, C=4) gets the full
    # (pattern, n_grid) sweep because all three channels exercise the delta
    # logic; other combos get representative configs to confirm channel
    # subsets integrate.
    # --- default combo: full sweep ---
    ("u_chi_f", 4, "all_moved", 1),
    ("u_chi_f", 4, "all_moved", 3),
    ("u_chi_f", 4, "all_moved", 6),
    ("u_chi_f", 4, "none_moved", 1),
    ("u_chi_f", 4, "none_moved", 6),
    ("u_chi_f", 4, "mixed", 3),
    ("u_chi_f", 4, "mixed", 6),
    # --- C=3 of default combo: confirm cutoff exponent variation ---
    ("u_chi_f", 3, "all_moved", 3),
    ("u_chi_f", 3, "mixed", 6),
    # --- channel-subset combos: representative configs ---
    ("u", 4, "all_moved", 3),
    ("u", 3, "mixed", 6),
    ("u_chi", 4, "all_moved", 6),
    ("u_chi", 3, "none_moved", 1),
    ("u_f", 4, "all_moved", 3),
    ("u_f", 3, "mixed", 6),
]


def _apply_move_pattern_rank1(
    new_r_up_carts_arr: np.ndarray,
    new_r_dn_carts_arr: np.ndarray,
    pattern: str,
    n_up: int,
    n_dn: int,
) -> None:
    """Apply the J3-style move pattern to the proposed-grid arrays in place.

    Mirrors the pattern logic in ``test_ratio_Jastrow_part_rank1_update_*``
    (tests/test_jqmc_kernel/test_jastrow.py): ``all_moved`` rotates (spin, index, axis) per
    grid; ``none_moved`` leaves every grid at the old positions; ``mixed``
    only moves on the first (and last, when ``n_grid>=3``) grids.
    """
    n_grid = new_r_up_carts_arr.shape[0]
    up_alt_idx = min(1, n_up - 1) if n_up > 0 else 0
    dn_alt_idx = min(1, n_dn - 1) if n_dn > 0 else 0

    if pattern == "all_moved":
        for g in range(n_grid):
            r = g % 3
            if r == 0 and n_up > 0:
                new_r_up_carts_arr[g, 0, 0] += 0.05
            elif r == 1 and n_dn > 0:
                new_r_dn_carts_arr[g, 0, 1] -= 0.07
            elif n_up > 0:
                new_r_up_carts_arr[g, up_alt_idx, 2] += 0.09
            elif n_dn > 0:
                new_r_dn_carts_arr[g, dn_alt_idx, 0] += 0.06
    elif pattern == "none_moved":
        return
    elif pattern == "mixed":
        if n_up > 0:
            new_r_up_carts_arr[0, 0, 0] += 0.05
        if n_grid >= 3 and n_dn > 0:
            new_r_dn_carts_arr[n_grid - 1, dn_alt_idx, 1] += 0.04
    else:
        raise ValueError(f"unknown pattern: {pattern}")


def _run_dtn_rank1_check(sd, preset, C, pattern, n_grid):
    atol, rtol = get_tolerance("jastrow_ratio", "strict")
    dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=1)
    jastrow_data = Jastrow_data(jastrow_dtn_data=dtn)
    r_up, r_dn = _draw_electrons(sd, 2, 1, seed=2)

    new_r_up_carts_arr = np.repeat(r_up[None, ...], n_grid, axis=0)
    new_r_dn_carts_arr = np.repeat(r_dn[None, ...], n_grid, axis=0)
    _apply_move_pattern_rank1(new_r_up_carts_arr, new_r_dn_carts_arr, pattern, r_up.shape[0], r_dn.shape[0])

    ratio_debug = _compute_ratio_Jastrow_part_debug(
        jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )
    ratio_fast = _compute_ratio_Jastrow_part_rank1_update(
        jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
    )
    assert ratio_fast.shape == (n_grid,)
    np.testing.assert_allclose(np.asarray(ratio_fast), np.asarray(ratio_debug), atol=atol, rtol=rtol)
    if pattern == "none_moved":
        np.testing.assert_allclose(np.asarray(ratio_debug), np.ones((n_grid,), dtype=np.float64), atol=atol, rtol=rtol)


# Split-spin form is used by the LRDMC projection grid and ECP nonlocal
# nearest-neighbor path: each block carries moves of only one spin so the
# block dispatch is direct (no per-config jnp.where). Test mirrors the
# rank-1 sweep above but for the ``_compute_ratio_Jastrow_part_split_spin``
# API surface.
_RATIO_SPLIT_DTN_PARAM_SETS = [
    # (preset, C, g_up, g_dn) -- patterns are folded into the move-loop below
    # (all blocks rotate (idx, axis) per grid).
    ("u_chi_f", 4, 1, 1),
    ("u_chi_f", 4, 3, 3),
    ("u_chi_f", 4, 6, 6),
    ("u_chi_f", 4, 6, 1),
    ("u_chi_f", 4, 1, 6),
    ("u_chi_f", 3, 3, 3),
    ("u", 4, 3, 3),
    ("u_chi", 4, 6, 3),
    ("u_f", 4, 3, 6),
    ("u_f", 3, 6, 6),
]


def _run_dtn_split_spin_check(sd, preset, C, g_up, g_dn):
    atol, rtol = get_tolerance("jastrow_ratio", "strict")
    dtn = _build_dtn(sd, preset=preset, cutoff_smoothness=C, seed=1)
    jastrow_data = Jastrow_data(jastrow_dtn_data=dtn)
    r_up, r_dn = _draw_electrons(sd, 2, 1, seed=2)

    new_up_shifted = np.repeat(r_up[None], g_up, axis=0)
    new_dn_shifted = np.repeat(r_dn[None], g_dn, axis=0)
    for g in range(g_up):
        new_up_shifted[g, g % r_up.shape[0], g % 3] += 0.04 * (g + 1)
    for g in range(g_dn):
        new_dn_shifted[g, g % r_dn.shape[0], (g + 1) % 3] += 0.03 * (g + 1)

    ratio_split = _compute_ratio_Jastrow_part_split_spin(
        jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_shifted=new_up_shifted,
        new_r_dn_shifted=new_dn_shifted,
    )
    combined_up = np.concatenate(
        [new_up_shifted, np.broadcast_to(r_up[None], (g_dn, r_up.shape[0], 3))],
        axis=0,
    )
    combined_dn = np.concatenate(
        [np.broadcast_to(r_dn[None], (g_up, r_dn.shape[0], 3)), new_dn_shifted],
        axis=0,
    )
    ratio_brute = _compute_ratio_Jastrow_part_debug(
        jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=combined_up,
        new_r_dn_carts_arr=combined_dn,
    )
    assert ratio_split.shape == (g_up + g_dn,)
    np.testing.assert_allclose(np.asarray(ratio_split), np.asarray(ratio_brute), atol=atol, rtol=rtol)


class TestDTNRatioConsistency:
    """Ratio consistency tests (rank-1 / split-spin vs brute force).

    Covers both the single-pbc-toggle smoke tests
    (``_molecular_or_periodic``) and the full (preset x C x pattern x
    n_grid) parametric sweeps for the rank-1 fast update and the
    split-spin form.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    def test_ratio_Jastrow_dtn_rank1_molecular_or_periodic(self, pbc):
        """Single-electron move ratio via rank1_update agrees with brute force."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=1)
        jastrow_data = Jastrow_data(jastrow_dtn_data=dtn)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=2)

        n_grid = 5
        new_up = np.repeat(r_up[None], n_grid, axis=0)
        new_dn = np.repeat(r_dn[None], n_grid, axis=0)
        for g in range(n_grid):
            spin = "up" if g % 2 == 0 else "dn"
            idx = g % max(r_up.shape[0], 1) if spin == "up" else g % max(r_dn.shape[0], 1)
            axis = g % 3
            shift = 0.03 * (1 + 0.1 * g)
            if spin == "up":
                new_up[g, idx, axis] += shift
            else:
                new_dn[g, idx, axis] += shift

        ratio_brute = _compute_ratio_Jastrow_part_debug(
            jastrow_data,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_carts_arr=new_up,
            new_r_dn_carts_arr=new_dn,
        )
        ratio_rank1 = _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_carts_arr=new_up,
            new_r_dn_carts_arr=new_dn,
        )
        np.testing.assert_allclose(np.asarray(ratio_rank1), np.asarray(ratio_brute), atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    def test_ratio_Jastrow_dtn_split_spin_molecular_or_periodic(self, pbc):
        """split_spin ratio with DTN agrees with brute force."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=1)
        jastrow_data = Jastrow_data(jastrow_dtn_data=dtn)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=2)

        g_up = 3
        g_dn = 2
        new_up_shifted = np.repeat(r_up[None], g_up, axis=0)
        new_dn_shifted = np.repeat(r_dn[None], g_dn, axis=0)
        for g in range(g_up):
            new_up_shifted[g, g % r_up.shape[0], g % 3] += 0.04 * (g + 1)
        for g in range(g_dn):
            new_dn_shifted[g, g % r_dn.shape[0], (g + 1) % 3] += 0.03 * (g + 1)

        ratio_split = _compute_ratio_Jastrow_part_split_spin(
            jastrow_data,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_shifted=new_up_shifted,
            new_r_dn_shifted=new_dn_shifted,
        )
        # Brute-force reference: build a concatenated (G_up + G_dn) array where
        # the first G_up entries hold up-block moves and the rest dn-block moves.
        combined_up = np.concatenate(
            [new_up_shifted, np.broadcast_to(r_up[None], (g_dn, r_up.shape[0], 3))],
            axis=0,
        )
        combined_dn = np.concatenate(
            [np.broadcast_to(r_dn[None], (g_up, r_dn.shape[0], 3)), new_dn_shifted],
            axis=0,
        )
        ratio_brute = _compute_ratio_Jastrow_part_debug(
            jastrow_data,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_carts_arr=combined_up,
            new_r_dn_carts_arr=combined_dn,
        )
        np.testing.assert_allclose(np.asarray(ratio_split), np.asarray(ratio_brute), atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("preset,C,pattern,n_grid", _RATIO_RANK1_DTN_PARAM_SETS)
    def test_ratio_Jastrow_dtn_rank1_sweep_molecular(self, preset, C, pattern, n_grid):
        """Rank-1 fast update vs brute-force ratio for DTN -- molecular sweep."""
        jax.clear_caches()
        sd = _h2_structure()
        _run_dtn_rank1_check(sd, preset, C, pattern, n_grid)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("preset,C,pattern,n_grid", _RATIO_RANK1_DTN_PARAM_SETS)
    def test_ratio_Jastrow_dtn_rank1_sweep_periodic(self, preset, C, pattern, n_grid):
        """Rank-1 fast update vs brute-force ratio for DTN -- PBC sweep."""
        jax.clear_caches()
        sd = _h2_pbc_structure()
        _run_dtn_rank1_check(sd, preset, C, pattern, n_grid)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("preset,C,g_up,g_dn", _RATIO_SPLIT_DTN_PARAM_SETS)
    def test_ratio_Jastrow_dtn_split_spin_sweep_molecular(self, preset, C, g_up, g_dn):
        """Split-spin fast update vs brute-force ratio for DTN -- molecular sweep."""
        jax.clear_caches()
        sd = _h2_structure()
        _run_dtn_split_spin_check(sd, preset, C, g_up, g_dn)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("preset,C,g_up,g_dn", _RATIO_SPLIT_DTN_PARAM_SETS)
    def test_ratio_Jastrow_dtn_split_spin_sweep_periodic(self, preset, C, g_up, g_dn):
        """Split-spin fast update vs brute-force ratio for DTN -- PBC sweep."""
        jax.clear_caches()
        sd = _h2_pbc_structure()
        _run_dtn_split_spin_check(sd, preset, C, g_up, g_dn)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Position gradient, HDF5, Coulomb-fast, converter and parameter-validation tests
# ---------------------------------------------------------------------------


# Helpers / fixtures (QE convert-to wiring) -- module-level so the
# ``@pytest.mark.skipif(not _LI_QE_AVAILABLE, ...)`` decorators below can
# resolve them at import time.
_LI_QE_FIXTURE = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_LI_QE_AVAILABLE = (_LI_QE_FIXTURE / "data-file-schema.xml").is_file() and (_LI_QE_FIXTURE / "wfc1.hdf5").is_file()
_LI_UPF = _LI_QE_FIXTURE / "Li.ccECP.upf"


def _qe_convert_to_common_kwargs(out_file: Path) -> dict:
    """Shared keyword arguments for ``qe_convert_to`` programmatic calls.

    The typer-style ``OptionInfo`` defaults need to be overridden explicitly
    when the function is invoked programmatically (the same caveat already
    documented in ``test_qe_convert_to_ecutwfc_auto_end_to_end``).
    """
    return dict(
        qe_save_dir=str(_LI_QE_FIXTURE),
        kind="pgto",
        mesh_factor=1.0,
        spline_order=5,
        upf_paths=[f"Li={_LI_UPF}"],
        ecutwfc=None,
        alpha_min=None,
        max_cond_number=None,
        hamiltonian_file=str(out_file),
        j1_parameter=None,
        j1_type="exp",
        j2_parameter=None,  # disable Pade J2 so we can attach J3 or DTN cleanly
        j2_type="pade",
        ewald_tol=1.0e-10,
        ecp_z_cores=None,
        verbosity=None,
    )


class TestDTNPhaseB:
    """Position-grad FD, HDF5 round-trip, Coulomb-fast with
    DTN, TREXIO + QE convert-to wiring, variational block round-trip, and
    parameter-validation rejection paths.
    """

    # Force consistency: autodiff vs FD wrt ion positions
    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(reason="FD / numerical quadrature comparison dominated by float32 round-off")
    @pytest.mark.parametrize("pbc", [False, True])
    def test_dtn_position_grad_matches_fd(self, pbc):
        """Autodiff w.r.t. ion positions equals central FD of ``compute_Jastrow_DTN``.

        The DTN evaluator depends on ion positions through the chi (e-n) and
        f (e-e-n) cutoffs; this test confirms that the AD gradient threaded
        through ``Structure_data.positions`` (the path used by both MCMC's
        ``accumulate_position_grad`` and the GFMC inline force line) matches
        a finite-difference reference on a small fixture.
        """
        import jax.numpy as jnp

        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=4, L=2.5)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=5)

        def J_at(positions):
            # ``_positions_cart_jnp`` is a property derived from ``positions`` so
            # replacing the latter is enough; AD then flows naturally through the
            # property at trace time.
            new_sd = sd.replace(positions=positions)
            new_dtn = dtn.replace(structure_data=new_sd)
            return compute_Jastrow_DTN(new_dtn, r_up, r_dn)

        base_positions = jnp.asarray(sd.positions, dtype=jnp.float64)
        grad_ad = jax.grad(J_at)(base_positions)
        n_atom, _ = np.asarray(sd.positions).shape

        h = 1e-5
        grad_fd = np.zeros((n_atom, 3), dtype=np.float64)
        base_np = np.asarray(sd.positions, dtype=np.float64)
        for I in range(n_atom):
            for d in range(3):
                pos_p = base_np.copy()
                pos_m = base_np.copy()
                pos_p[I, d] += h
                pos_m[I, d] -= h
                J_p = float(J_at(jnp.asarray(pos_p)))
                J_m = float(J_at(jnp.asarray(pos_m)))
                grad_fd[I, d] = (J_p - J_m) / (2 * h)

        np.testing.assert_allclose(np.asarray(grad_ad), grad_fd, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT)
        record_fd_case(
            "fd-ad-dtn-ion-position",
            pytest_node_id="tests/test_jqmc_kernel/test_jastrow_dtn.py::TestDTNPhaseB::test_dtn_position_grad_matches_fd",
            fixture_parameters=fd_case_parameters("fd-ad-dtn-ion-position", pbc=pbc),
            coordinates=(r_up, r_dn),
            coordinate_generator="tests.test_jqmc_kernel.test_jastrow_dtn._draw_electrons",
            coordinate_seed=5,
            compared_pairs=((grad_ad, grad_fd),),
            atol=ATOL_FD_VS_ANALYTIC_TIGHT,
            rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
            zones=(),
            level=None,
            comparison_policy_id="CMP-FD-AUTODIFF-FULL-001",
            stencil="central-first-order-2",
            step=1.0e-5,
            direction="cartesian-basis",
        )
        jax.clear_caches()

    # HDF5 round-trip
    @pytest.mark.light
    def test_jastrow_dtn_hdf5_roundtrip(self, tmp_path):
        """``Jastrow_DTN_data`` round-trips through the generic HDF5 walker.

        Uses ``_save_dataclass_to_hdf5`` / ``_load_dataclass_from_hdf5`` -- the
        same machinery that ``Hamiltonian_data.save_to_hdf5`` rides on -- and
        confirms the post-load value of ``compute_Jastrow_DTN`` is bit-identical
        to the pre-save value. The constraint inflation matrices ``M_*`` are
        persisted as plain arrays via the generic walker (no
        DTN-specific backfill needed).
        """
        import h5py

        from jqmc.hamiltonians import _load_dataclass_from_hdf5, _save_dataclass_to_hdf5

        sd = _h2_pbc_structure()
        data = _build_dtn(sd, preset="u_chi_f", seed=21)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=22)
        J_pre = float(compute_Jastrow_DTN(data, r_up, r_dn))

        path = tmp_path / "dtn.h5"
        with h5py.File(path, "w") as f:
            _save_dataclass_to_hdf5(f.create_group("dtn"), data)
        with h5py.File(path, "r") as f:
            loaded = _load_dataclass_from_hdf5(Jastrow_DTN_data, f["dtn"])
        with h5py.File(path, "r") as f:
            loaded_again = _load_dataclass_from_hdf5(Jastrow_DTN_data, f["dtn"])

        legacy_path = tmp_path / "legacy_dtn.h5"
        with h5py.File(legacy_path, "w") as f:
            _save_dataclass_to_hdf5(f.create_group("dtn"), data.replace(apply_en_cusp=False))
        with h5py.File(legacy_path, "r") as f:
            loaded_legacy = _load_dataclass_from_hdf5(Jastrow_DTN_data, f["dtn"])
        assert loaded_legacy.apply_en_cusp is False
        with pytest.raises(ValueError, match=r"raw local energy.*357518.*VMC optimization"):
            loaded_legacy.sanity_check()

        # Variational arrays round-tripped exactly.
        np.testing.assert_array_equal(np.asarray(loaded.alpha_indep), np.asarray(data.alpha_indep))
        np.testing.assert_array_equal(np.asarray(loaded.beta_indep), np.asarray(data.beta_indep))
        np.testing.assert_array_equal(np.asarray(loaded.gamma_indep), np.asarray(data.gamma_indep))
        np.testing.assert_array_equal(np.asarray(loaded.L_u), np.asarray(data.L_u))
        np.testing.assert_array_equal(np.asarray(loaded.L_chi), np.asarray(data.L_chi))
        np.testing.assert_array_equal(np.asarray(loaded.L_f), np.asarray(data.L_f))
        assert loaded.cutoff_ratio_u == data.cutoff_ratio_u
        assert loaded.cutoff_ratio_chi == data.cutoff_ratio_chi
        assert loaded.cutoff_ratio_f == data.cutoff_ratio_f
        np.testing.assert_array_equal(loaded.policy_T_alpha, data.policy_T_alpha)
        np.testing.assert_array_equal(loaded.policy_q_alpha, data.policy_q_alpha)
        np.testing.assert_array_equal(loaded.policy_T_beta, data.policy_T_beta)
        np.testing.assert_array_equal(loaded.policy_q_beta, data.policy_q_beta)
        np.testing.assert_array_equal(loaded.policy_T_gamma, data.policy_T_gamma)
        np.testing.assert_array_equal(loaded.policy_q_gamma, data.policy_q_gamma)
        # Static metadata preserved.
        assert loaded.use_u == data.use_u
        assert loaded.use_chi == data.use_chi
        assert loaded.use_f == data.use_f
        assert loaded.apply_en_cusp == data.apply_en_cusp
        assert loaded.cutoff_smoothness == data.cutoff_smoothness
        # Inflation matrices round-tripped.
        np.testing.assert_array_equal(loaded.M_alpha, data.M_alpha)
        np.testing.assert_array_equal(loaded.b_alpha, data.b_alpha)
        np.testing.assert_array_equal(loaded.M_gamma, data.M_gamma)

        # Every level of tuple-annotated static metadata must remain a tuple,
        # never an HDF5-produced list / ndarray.  Value-only round-trip checks
        # do not catch this: JAX compares static aux data only when a second
        # same-structure object enters an existing JIT cache slot.
        def _assert_recursively_hashable_tuple(value):
            assert isinstance(value, tuple)
            for item in value:
                if isinstance(item, tuple):
                    _assert_recursively_hashable_tuple(item)
                else:
                    assert not isinstance(item, (list, np.ndarray))

        for field in ("M_alpha", "b_alpha", "M_beta", "b_beta", "M_gamma"):
            _assert_recursively_hashable_tuple(getattr(loaded, field))
            hash(getattr(loaded, field))

        # Regression for the production failure: the first loaded DTN traces;
        # the second forces JAX to compare the cached pytree metadata.  Before
        # recursive tuple restoration this raises "arrays cannot be passed as
        # metadata fields" even though either call succeeds in isolation.
        jax.clear_caches()
        grad_lap_1 = compute_grads_and_laplacian_Jastrow_DTN(loaded, r_up, r_dn)
        grad_lap_2 = compute_grads_and_laplacian_Jastrow_DTN(loaded_again, r_up, r_dn)
        for actual, expected in zip(grad_lap_2, grad_lap_1):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        # Post-load production value matches.
        J_post = float(compute_Jastrow_DTN(loaded, r_up, r_dn))
        np.testing.assert_allclose(J_post, J_pre, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)
        jax.clear_caches()

    # Fast-update Coulomb coexistence with DTN
    @pytest.mark.light
    def test_compute_coulomb_potential_fast_with_dtn(self):
        """compute_coulomb_potential_fast still matches the reference with DTN present.

        The fast path goes through ``_compute_ratio_Jastrow_part_*`` for the
        non-local ECP Psi(r')/Psi(r) ratios; ``WF_ratio = J_ratio * Det_ratio``
        is a product, so the fast path inherits
        DTN correctness automatically once the rank-1 + split-spin paths
        agree with the brute-force reference (the ratio-consistency tests).
        This test runs the actual fast vs reference comparison with DTN
        populated to guard the dispatch chain.
        """
        import os

        import jax.numpy as jnp

        from jqmc._precision import get_tolerance
        from jqmc.coulomb_potential import compute_coulomb_potential, compute_coulomb_potential_fast
        from jqmc.determinant import compute_geminal_all_elements
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = "water_ccecp_ccpvqz.h5"
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", fixture)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {fixture} is not present.")
        atol, rtol = get_tolerance("coulomb", "strict")

        structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )
        dtn = _build_dtn(structure_data, preset="u_chi_f", seed=7, L=2.0)
        wavefunction_data = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(jastrow_dtn_data=dtn),
        )

        rng = np.random.default_rng(0)
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        r_up_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_up, 3)))
        r_dn_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_dn, 3)))
        G = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        A_old_inv = jnp.linalg.inv(G)

        RT = jnp.eye(3)
        Nv = 6
        V_ref = compute_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            RT=RT,
            wavefunction_data=wavefunction_data,
            Nv=Nv,
        )
        V_fast = compute_coulomb_potential_fast(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            RT=RT,
            A_old_inv=A_old_inv,
            wavefunction_data=wavefunction_data,
            Nv=Nv,
        )
        np.testing.assert_allclose(np.asarray(V_fast), np.asarray(V_ref), atol=atol, rtol=rtol)
        jax.clear_caches()

    # CLI: --jastrow-dtn end-to-end via trexio convert-to
    @pytest.mark.light
    def test_trexio_convert_to_with_dtn(self, tmp_path):
        """trexio convert-to with --jastrow-dtn=u_chi_f writes a DTN-only Hamiltonian.

        Smoke-tests the CLI wiring end-to-end:
          * the convert command accepts the new flag,
          * the resulting Hamiltonian has ``jastrow_dtn_data`` populated and
            ``jastrow_one_body_data`` / ``jastrow_two_body_data`` /
            ``jastrow_three_body_data`` all None,
          * round-trips through HDF5.
        """
        import os

        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_tool import trexio_convert_to

        fixture = "water_ccecp_ccpvqz.h5"
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", fixture)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {fixture} is not present.")

        out_path = tmp_path / "ham_dtn.h5"
        # Call the typer function directly (programmatic invocation; typer
        # OptionInfo defaults are tolerated by the function body).
        # Programmatic invocation: pass None for each Pade/AO Jastrow option so
        # the build skips the legacy family (typer's OptionInfo defaults would
        # otherwise leak in as truthy).
        trexio_convert_to(
            trexio_file=trexio_path,
            hamiltonian_file=str(out_path),
            j1_parmeter=None,
            j1_type="exp",
            j2_parmeter=None,
            j2_type="pade",
            j3_basis_type="none",
            jastrow_dtn="u_chi_f",
            dtn_c=4,
            dtn_n_u=4,
            dtn_n_chi=4,
            dtn_n_eN=2,
            dtn_n_ee=2,
            dtn_L_u=2.5,
            dtn_L_chi=2.5,
            dtn_L_f=2.5,
            dtn_cutoff_abs_cap=8.0,
            ao_conv_to=None,
        )
        assert out_path.exists()

        ham = Hamiltonian_data.load_from_hdf5(str(out_path))
        jd = ham.wavefunction_data.jastrow_data
        assert jd.jastrow_dtn_data is not None
        assert jd.jastrow_one_body_data is None
        assert jd.jastrow_two_body_data is None
        assert jd.jastrow_three_body_data is None
        assert jd.jastrow_dtn_data.use_u is True
        assert jd.jastrow_dtn_data.use_chi is True
        assert jd.jastrow_dtn_data.use_f is True
        legacy_jastrow = jd.replace(jastrow_dtn_data=jd.jastrow_dtn_data.replace(apply_en_cusp=False))
        legacy_ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=legacy_jastrow))
        with pytest.raises(ValueError, match=r"raw local energy.*357518.*VMC optimization"):
            legacy_ham.sanity_check()
        jax.clear_caches()

    # Variational block round-trip.
    @pytest.mark.light
    def test_dtn_variational_blocks_round_trip(self):
        """``get_variational_blocks`` -> ``apply_block_updates`` round-trips DTN.

        Validates that the wavefunction layer correctly emits ``j3_poly_alpha`` /
        ``j3_poly_beta`` / ``j3_poly_gamma`` blocks for a DTN-only Jastrow and
        that ``apply_block_updates`` propagates a flat theta vector back into the
        independent-coefficient arrays without losing the cusp constraints
        (which are baked into ``M_alpha`` / ``b_alpha`` / ``M_beta`` /
        ``b_beta`` / ``M_gamma``, all ``pytree_node=False``).
        """
        from jqmc.wavefunction import Wavefunction_data

        sd = _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=33, L=2.0, N_u=4, N_chi=4, N_eN=2, N_ee=2)
        wf = Wavefunction_data(geminal_data=None, jastrow_data=Jastrow_data(jastrow_dtn_data=dtn))

        blocks = wf.get_variational_blocks(
            state_token=("test", 0),
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
        )
        names = [b.name for b in blocks]
        assert "j3_poly_alpha" in names
        assert "j3_poly_beta" in names
        assert "j3_poly_gamma" in names
        sizes = {b.name: b.size for b in blocks}
        assert sizes["j3_poly_alpha"] == dtn.alpha_indep.size
        assert sizes["j3_poly_beta"] == dtn.beta_indep.size
        assert sizes["j3_poly_gamma"] == dtn.gamma_indep.size

        total = sum(b.size for b in blocks)
        thetas = np.linspace(0.05, 0.15, total, dtype=np.float64)
        wf_new = wf.apply_block_updates(blocks, thetas, learning_rate=1.0)
        new_dtn = wf_new.jastrow_data.jastrow_dtn_data
        # Each DTN array must have shifted by the corresponding theta slice.
        np.testing.assert_array_equal(new_dtn.alpha_indep.shape, dtn.alpha_indep.shape)
        np.testing.assert_array_equal(new_dtn.beta_indep.shape, dtn.beta_indep.shape)
        np.testing.assert_array_equal(new_dtn.gamma_indep.shape, dtn.gamma_indep.shape)
        # The constraint inflation matrices must be carried through unchanged
        # (they describe the structure of the cusp algebra, not variational data).
        np.testing.assert_array_equal(new_dtn.M_alpha, dtn.M_alpha)
        np.testing.assert_array_equal(new_dtn.M_beta, dtn.M_beta)
        np.testing.assert_array_equal(new_dtn.M_gamma, dtn.M_gamma)
        np.testing.assert_array_equal(new_dtn.b_alpha, dtn.b_alpha)
        np.testing.assert_array_equal(new_dtn.b_beta, dtn.b_beta)
        # And the post-update value of the e-e cusp must still be exactly 1/2
        # for the antiparallel channel: alpha_full[1, 1] = (C/L) alpha[1,0] +
        # cusp_bias. Verify by re-deriving the inflated alpha and checking the
        # cusp linear-coefficient.
        M_alpha = np.asarray(new_dtn.M_alpha)
        b_alpha = np.asarray(new_dtn.b_alpha)
        alpha_indep_new = np.asarray(new_dtn.alpha_indep)
        alpha_full = np.einsum("ski,si->sk", M_alpha, alpha_indep_new) + b_alpha
        L = float(new_dtn.L_u)
        C = new_dtn.cutoff_smoothness
        # u'_anti(0+) = L^{C-1} * (L alpha_1 - C alpha_0)  should equal 1/2 ((L - r)^C form).
        cusp = (L ** (C - 1)) * (L * alpha_full[1, 1] - C * alpha_full[1, 0])
        np.testing.assert_allclose(cusp, 0.5, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        jax.clear_caches()

    # QE convert-to wiring for J3 + DTN
    @pytest.mark.light
    @pytest.mark.skipif(not _LI_QE_AVAILABLE, reason="tests/qe_example_files/Li_gamma not present")
    def test_qe_convert_to_with_dtn(self, tmp_path):
        """QE convert-to writes a DTN Hamiltonian when ``--jastrow-dtn`` is set."""
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_tool import qe_convert_to

        out_path = tmp_path / "li_dtn.h5"
        kwargs = _qe_convert_to_common_kwargs(out_path)
        qe_convert_to(
            **kwargs,
            j3_basis_type="none",
            jastrow_dtn="u_chi_f",
            dtn_c=4,
            dtn_n_u=4,
            dtn_n_chi=4,
            dtn_n_eN=2,
            dtn_n_ee=2,
            # Auto-resolved cutoffs (bcc Li conventional cell with alat ~ 6.63
            # Bohr has R_WS_smooth ~ a/2 / sqrt(2); auto picks 0.95 *
            # R_WS_smooth which fits comfortably below the wrap threshold).
            dtn_cutoff_ratio_u=0.75,
            dtn_cutoff_ratio_chi=0.75,
            dtn_cutoff_ratio_f=0.75,
        )
        assert out_path.exists()

        ham = Hamiltonian_data.load_from_hdf5(str(out_path))
        jd = ham.wavefunction_data.jastrow_data
        assert jd.jastrow_dtn_data is not None
        assert jd.jastrow_one_body_data is None
        assert jd.jastrow_two_body_data is None
        assert jd.jastrow_three_body_data is None
        assert jd.jastrow_dtn_data.use_u is True
        assert jd.jastrow_dtn_data.use_chi is True
        assert jd.jastrow_dtn_data.use_f is True
        # Li.ccECP has z_core=2 so z_eff = 3 - 2 = 1; the converter wires
        # this through to ``z_eff_per_species[0]`` -- assert positive (the
        # exact value follows from the ccECP table).
        assert jd.jastrow_dtn_data.z_eff_per_species[0] > 0
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skipif(not _LI_QE_AVAILABLE, reason="tests/qe_example_files/Li_gamma not present")
    def test_qe_convert_to_with_j3_ao(self, tmp_path):
        """QE convert-to writes a J3 (analytic AO) Hamiltonian when ``-j3 ao`` is set."""
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_tool import qe_convert_to

        out_path = tmp_path / "li_j3.h5"
        kwargs = _qe_convert_to_common_kwargs(out_path)
        # Re-enable the Pade J2 baseline alongside J3 (matches the typical
        # TREXIO convert workflow).
        kwargs["j2_parameter"] = 1.0
        qe_convert_to(
            **kwargs,
            j3_basis_type="ao",
            jastrow_dtn="none",
            dtn_c=4,
            dtn_n_u=8,
            dtn_n_chi=8,
            dtn_n_eN=4,
            dtn_n_ee=4,
            dtn_cutoff_ratio_u=0.75,
            dtn_cutoff_ratio_chi=0.75,
            dtn_cutoff_ratio_f=0.75,
        )
        assert out_path.exists()

        ham = Hamiltonian_data.load_from_hdf5(str(out_path))
        jd = ham.wavefunction_data.jastrow_data
        assert jd.jastrow_three_body_data is not None
        assert jd.jastrow_dtn_data is None
        assert jd.jastrow_two_body_data is not None  # Pade J2 baseline
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skipif(not _LI_QE_AVAILABLE, reason="tests/qe_example_files/Li_gamma not present")
    def test_qe_convert_to_rejects_dtn_with_pade_family(self, tmp_path):
        """``--jastrow-dtn != none`` together with -j1/-j2/-j3 must raise."""
        import click as _click

        from jqmc.jqmc_tool import qe_convert_to

        out_path = tmp_path / "should_not_exist.h5"
        kwargs = _qe_convert_to_common_kwargs(out_path)
        kwargs["j2_parameter"] = 1.0  # Pade J2 enabled
        with pytest.raises(_click.ClickException, match="mutually exclusive"):
            qe_convert_to(
                **kwargs,
                j3_basis_type="none",
                jastrow_dtn="u_chi_f",
                dtn_c=4,
                dtn_n_u=4,
                dtn_n_chi=4,
                dtn_n_eN=2,
                dtn_n_ee=2,
                dtn_cutoff_ratio_u=0.75,
                dtn_cutoff_ratio_chi=0.75,
                dtn_cutoff_ratio_f=0.75,
            )
        assert not out_path.exists()


# ---------------------------------------------------------------------------
# Molecular DTN cutoffs are lengths (issue #589): a cutoff L <= 0 (or a
# non-finite one) makes the (L - r)^C factor vanish for every r > 0, i.e. it
# silently removes the requested channel and its Kato cusp. The converter,
# the factory and the load-time sanity check must refuse it.
# ---------------------------------------------------------------------------

#: case -> (extra ``jqmc-tool trexio convert-to`` arguments, expected message fragment).
_NONPOSITIVE_DTN_CUTOFF_CLI_CASES = {
    "Lu-negative": (["--dtn-Lu", "-1"], "--dtn-Lu = -1.0 must be a finite cutoff length > 0"),
    "Lu-zero": (["--dtn-Lu", "0"], "--dtn-Lu = 0.0 must be a finite cutoff length > 0"),
    "Lu-nan": (["--dtn-Lu", "nan"], "--dtn-Lu = nan must be a finite cutoff length > 0"),
    "Lchi-negative": (["--dtn-Lchi", "-1"], "--dtn-Lchi = -1.0 must be a finite cutoff length > 0"),
    "Lchi-zero": (["--dtn-Lchi", "0"], "--dtn-Lchi = 0.0 must be a finite cutoff length > 0"),
    "Lf-negative": (["--dtn-Lf", "-1"], "--dtn-Lf = -1.0 must be a finite cutoff length > 0"),
    "Lf-zero": (["--dtn-Lf", "0"], "--dtn-Lf = 0.0 must be a finite cutoff length > 0"),
    "cap-negative": (["--dtn-cutoff-abs-cap", "-1"], "--dtn-cutoff-abs-cap = -1.0 must be a finite cutoff length > 0"),
    "cap-zero": (["--dtn-cutoff-abs-cap", "0"], "--dtn-cutoff-abs-cap = 0.0 must be a finite cutoff length > 0"),
}

#: case -> (``Jastrow_DTN_data.build`` keyword overrides, expected message fragment).
_NONPOSITIVE_DTN_CUTOFF_BUILD_CASES = {
    "L_u-negative": ({"L_u": -1.0}, "L_u = -1.0 must be a finite length > 0"),
    "L_u-zero": ({"L_u": 0.0}, "L_u = 0.0 must be a finite length > 0"),
    "L_u-inf": ({"L_u": np.inf}, "L_u = inf must be a finite length > 0"),
    "L_chi-negative": ({"L_chi": -1.0}, r"L_chi = [-1.0] must be a finite length > 0"),
    "L_f-zero": ({"L_f": 0.0}, r"L_f = [0.0] must be a finite length > 0"),
    "cutoff_abs_cap-negative": ({"cutoff_abs_cap": -1.0}, "cutoff_abs_cap = -1.0 must be a finite length > 0"),
}

#: case -> (``Jastrow_DTN_data`` field overrides of a valid H2 object, expected message fragment).
_NONPOSITIVE_DTN_CUTOFF_STORED_CASES = {
    "L_u-negative": ({"L_u": -1.0}, "L_u = [-1.0] must contain finite lengths > 0"),
    "L_chi-negative": ({"L_chi": (-1.0,)}, "L_chi = [-1.0] must contain finite lengths > 0"),
    "L_f-zero": ({"L_f": (0.0,)}, "L_f = [0.0] must contain finite lengths > 0"),
    "L_u-nan": ({"L_u": float("nan")}, "L_u = [nan] must contain finite lengths > 0"),
}

_H2_AE_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")


def _one_line(text: str) -> str:
    """Join a CLI error that a rich-formatted box wrapped over several lines."""
    return " ".join(re.sub(r"[\u2500-\u257f]", " ", text).split())


def _h2_coalescence_slopes(data: Jastrow_DTN_data) -> np.ndarray:
    """Antiparallel e-e and all-electron e-n slopes of the production DTN gradient on H2.

    One electron moves along ``+x`` through a coalescence point at offsets
    ``+-h``; ``D(h) = (g(h) - g(-h)) / 2`` of its x-gradient cancels every
    term that is smooth there and leaves ``slope + O(h)``, and Richardson over
    ``h, h/2, h/4`` removes ``O(h)`` and ``O(h^2)`` (as ``_cusp_probe_slopes``).
    Offsets are powers of two, exact in fp32 and fp64.
    """
    nucleus = np.asarray(data.structure_data.positions, dtype=np.float64)[0]
    pair = nucleus + np.array([0.0, 0.5, 0.0])
    x_hat = np.array([1.0, 0.0, 0.0])

    def x_gradient(spin: str, origin: np.ndarray, other: np.ndarray, offset: float) -> float:
        moved = origin + offset * x_hat
        r_up, r_dn = (moved, other) if spin == "up" else (other, moved)
        g_up, g_dn, _l_up, _l_dn = compute_grads_and_laplacian_Jastrow_DTN(data, r_up[None, :], r_dn[None, :])
        grad = np.asarray(g_up if spin == "up" else g_dn, dtype=np.float64)
        assert np.all(np.isfinite(grad))
        return float(grad[0, 0])

    probes = (
        ("dn", pair, pair),  # down electron through the up electron: antiparallel e-e cusp
        ("up", nucleus, nucleus + np.array([0.0, -0.75, 0.0])),  # up electron through nucleus 0
    )
    slopes = []
    for spin, origin, other in probes:
        D = [
            0.5 * (x_gradient(spin, origin, other, h) - x_gradient(spin, origin, other, -h))
            for h in (_CUSP_PROBE_STEP, 0.5 * _CUSP_PROBE_STEP, 0.25 * _CUSP_PROBE_STEP)
        ]
        slopes.append((D[0] - 6.0 * D[1] + 8.0 * D[2]) / 3.0)
    return np.asarray(slopes)


class TestDTNMolecularCutoffValidation:
    """Issue #589: non-positive or non-finite molecular DTN cutoffs are refused."""

    @pytest.mark.light
    @pytest.mark.parametrize("case", list(_NONPOSITIVE_DTN_CUTOFF_CLI_CASES))
    def test_trexio_convert_to_rejects_nonpositive_dtn_cutoff(self, tmp_path, case):
        """``trexio convert-to --jastrow-dtn u_chi_f`` with a cutoff <= 0 or NaN exits non-zero and writes nothing.

        Accepting ``-1`` would give a Jastrow that is exactly 1 for every r > 0
        (the requested channel and its cusp gone), and ``0`` must not raise a raw
        ZeroDivisionError.
        """
        from click.testing import CliRunner

        from jqmc.jqmc_tool import _cli

        args, message = _NONPOSITIVE_DTN_CUTOFF_CLI_CASES[case]
        hamiltonian_file = tmp_path / "hamiltonian_data.h5"
        result = CliRunner().invoke(
            _cli,
            ["trexio", "convert-to", _H2_AE_TREXIO, "-o", str(hamiltonian_file), "--jastrow-dtn", "u_chi_f", *args],
        )
        assert result.exit_code != 0, result.output
        assert message in _one_line(result.output), (result.output, result.exception)
        assert "is saved" not in result.output
        assert not hamiltonian_file.exists()

    @pytest.mark.light
    @pytest.mark.parametrize("case", list(_NONPOSITIVE_DTN_CUTOFF_BUILD_CASES))
    def test_build_rejects_nonpositive_molecular_cutoff(self, case):
        """``Jastrow_DTN_data.build`` refuses a molecular cutoff (or auto-cutoff cap) that is not a positive length."""
        overrides, message = _NONPOSITIVE_DTN_CUTOFF_BUILD_CASES[case]
        kwargs = {"L_u": 3.0, "L_chi": 2.5, "L_f": 2.0, **overrides}
        with pytest.raises(ValueError, match=re.escape(message)):
            Jastrow_DTN_data.build(
                structure_data=_h2_structure(),
                z_eff_per_species=(1.0,),
                pseudized_per_species=(False,),
                n_order_u=4,
                n_order_chi=4,
                n_order_eN=2,
                n_order_ee=2,
                **kwargs,
            )

    @pytest.mark.light
    @pytest.mark.parametrize("case", list(_NONPOSITIVE_DTN_CUTOFF_STORED_CASES))
    def test_sanity_check_rejects_nonpositive_stored_cutoff(self, case):
        """The load-time ``sanity_check`` refuses stored cutoffs that are not positive lengths."""
        valid = _build_dtn(_h2_structure(), preset="u_chi_f", L=3.0)
        valid.sanity_check()
        overrides, message = _NONPOSITIVE_DTN_CUTOFF_STORED_CASES[case]
        with pytest.raises(ValueError, match=re.escape(message)):
            valid.replace(**overrides).sanity_check()

    @pytest.mark.light
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="explicit coalescence probes around the fixture's first nucleus",
    )
    def test_trexio_convert_to_valid_explicit_cutoffs_keep_kato_cusps(self, tmp_path):
        """Control: valid explicit cutoffs convert, are stored as requested and keep both Kato cusps.

        Independent reference: the antiparallel e-e cusp +1/2 and the
        all-electron hydrogen e-n cusp -Z = -1, measured as the jump of the
        production gradient across each coalescence point of the reloaded file.
        """
        from click.testing import CliRunner

        from jqmc.jqmc_tool import _cli

        hamiltonian_file = tmp_path / "hamiltonian_data.h5"
        cutoffs = {"--dtn-Lu": 3.0, "--dtn-Lchi": 2.5, "--dtn-Lf": 2.0}
        args = [arg for option, value in cutoffs.items() for arg in (option, str(value))]
        result = CliRunner().invoke(
            _cli,
            ["trexio", "convert-to", _H2_AE_TREXIO, "-o", str(hamiltonian_file), "--jastrow-dtn", "u_chi_f", *args],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        assert hamiltonian_file.is_file()

        dtn = Hamiltonian_data.load_from_hdf5(str(hamiltonian_file)).wavefunction_data.jastrow_data.jastrow_dtn_data
        assert (dtn.use_u, dtn.use_chi, dtn.use_f) == (True, True, True)
        assert float(np.asarray(dtn.L_u)) == cutoffs["--dtn-Lu"]
        assert [float(x) for x in dtn.L_chi] == [cutoffs["--dtn-Lchi"]]
        assert [float(x) for x in dtn.L_f] == [cutoffs["--dtn-Lf"]]

        slopes = _h2_coalescence_slopes(dtn)
        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        np.testing.assert_allclose(slopes, [0.5, -1.0], atol=atol, rtol=rtol, err_msg="(ee-antiparallel, en-all-electron)")
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Regression: 3rd-order AD (d/dr of the Laplacian == the ``d e_L / d r``
# path consumed by forces / stress / SWCT) must stay finite.
#
# Without the double-where guard in ``_dtn_scalar_dist_with_derivs``,
# ``sqrt(dot(u, u))`` carries a ``0/0`` gradient at the u-channel self-pair
# (``r_a == r_b``). Value / drift / Laplacian (<= 2nd order) stay finite (the
# self-pair is masked downstream), but differentiating the Laplacian once more
# (3rd order) leaks the NaN through the ``where``-mask (``0 * NaN``). Requires >= 2 same-spin electrons
# so the self-pair coexists with genuine pairs in the per-spin vmap -- a
# 1-up/1-dn system degenerates and hides the NaN. Both the molecular and the
# PBC branch are guarded.
# ---------------------------------------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize(
    "structure_factory, label",
    [(_h2_structure, "molecular"), (_h2_pbc_structure, "periodic")],
)
@pytest.mark.parametrize("preset", ["u", "u_chi_f"])
def test_dtn_third_order_ad_kinetic_derivative_finite_both_precisions(structure_factory, label, preset):
    """Both spin derivatives of the DTN kinetic surrogate stay finite in full and mixed.

    Use synthetic all-electron H2 structures (molecular or periodic), DTN
    u or u/chi/f with cutoff smoothness 3 and coefficient seed 3, and one
    configuration of two up and one down electrons from _draw_electrons
    with seed 3. These are the same four inputs as the full-only FD anchor
    test_dtn_third_order_ad_de_l_dr_is_finite. No TREXIO file, determinant,
    cond(G) screening, or stochastic acceptance is involved.

    Differentiate lap(J) + |grad(J)|^2 through the production DTN grad/lap
    evaluator, retaining the same-spin genuine pair alongside its masked
    self-pair. Require the entire up and down derivatives to be finite.
    One verdict requires both spin blocks, so a recorded passing assertion
    cannot stand for only one of the two checks.
    This is a NaN/Inf regression check, not a numerical-accuracy comparison;
    the independent finite-difference anchor remains full-only.
    """
    sd = structure_factory()
    data = _build_dtn(sd, preset=preset, cutoff_smoothness=3, seed=3)
    r_up_np, r_dn_np = _draw_electrons(sd, n_up=2, n_dn=1, seed=3)
    r_up = jnp.asarray(r_up_np)
    r_dn = jnp.asarray(r_dn_np)

    def kinetic_surrogate(ru, rd):
        g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_DTN(data, ru, rd)
        return jnp.sum(l_up) + jnp.sum(l_dn) + jnp.sum(g_up**2) + jnp.sum(g_dn**2)

    d_dr_up, d_dr_dn = jax.grad(kinetic_surrogate, argnums=(0, 1))(r_up, r_dn)
    assert jnp.all(jnp.isfinite(d_dr_up)) and jnp.all(jnp.isfinite(d_dr_dn)), (
        f"NaN/Inf in d e_L/d r_up or r_dn ({label}, {preset})"
    )


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="FD vs AD comparison dominated by float32 round-off",
    alternative_guarantee_ref="NUM-JAS-SMOKE-001",
)
@pytest.mark.parametrize(
    "structure_factory, label",
    [(_h2_structure, "molecular"), (_h2_pbc_structure, "periodic")],
)
@pytest.mark.parametrize("preset", ["u", "u_chi_f"])
def test_dtn_third_order_ad_de_l_dr_is_finite(structure_factory, label, preset):
    """DTN third order autodiff de l dr is finite."""
    sd = structure_factory()
    data = _build_dtn(sd, preset=preset, cutoff_smoothness=3, seed=3)
    # >= 2 same-spin electrons so the u-channel evaluates a self-pair (u == 0)
    # alongside genuine pairs -- the configuration that exposed the NaN.
    r_up_np, r_dn_np = _draw_electrons(sd, n_up=2, n_dn=1, seed=3)
    r_up = jnp.asarray(r_up_np)
    r_dn = jnp.asarray(r_dn_np)

    def kinetic_surrogate(ru, rd):
        # DTN contribution to the local kinetic energy is
        # ``-1/2 (lap J + |grad J|^2)``; ``d/dr`` of this is the 3rd-order path.
        g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_DTN(data, ru, rd)
        return jnp.sum(l_up) + jnp.sum(l_dn) + jnp.sum(g_up**2) + jnp.sum(g_dn**2)

    d_dr_up, d_dr_dn = jax.grad(kinetic_surrogate, argnums=(0, 1))(r_up, r_dn)
    assert jnp.all(jnp.isfinite(d_dr_up)), f"NaN/Inf in d e_L/d r_up ({label}, {preset})"
    assert jnp.all(jnp.isfinite(d_dr_dn)), f"NaN/Inf in d e_L/d r_dn ({label}, {preset})"

    # ... and it is the CORRECT derivative (central finite-difference oracle).
    def surrogate_up(ru):
        return float(kinetic_surrogate(jnp.asarray(ru), r_dn))

    h = 1e-5
    fd = np.zeros_like(r_up_np)
    for i in range(r_up_np.shape[0]):
        for d in range(3):
            rp = r_up_np.copy()
            rp[i, d] += h
            rm = r_up_np.copy()
            rm[i, d] -= h
            fd[i, d] = (surrogate_up(rp) - surrogate_up(rm)) / (2.0 * h)
    atol, rtol = get_tolerance("jastrow_grad_lap", "medium")
    np.testing.assert_allclose(np.asarray(d_dr_up), fd, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Regression: the ECP-nonlocal ``d e_L / d r`` and ``d e_L / d eps`` autograd
# (``_auto`` oracle) must stay FINITE (and match the analytic path) for a
# molecular ECP + DTN(u/chi/f) wavefunction with ONE electron per spin.
#
# The DTN ratio fast-update (``_delta_J_DTN_one_electron_moved``, fired by the
# ECP-nonlocal Jastrow ratio) evaluates the masked self pair
# ``r_fn(r_moved_old, r_moved_old) = 0`` (u same-spin block) and ``r_ij_old = 0``
# (f block) BEFORE the mask zeroes it. The molecular ``r_fn`` once used a bare
# ``jnp.linalg.norm`` whose ``0/0`` sqrt gradient survived the mask
# (``0 * NaN == NaN``) and NaN'd the ECP-nonlocal ``d e_L / d r`` / stress
# autograd -- with a SINGLE same-spin electron (the moved electron pairs with
# ITSELF), so unlike the kinetic-path 3rd-order regression above this needs NO
# ``>= 2`` same-spin pair. Guard the molecular ``r_fn`` with the same
# double-where the PBC branch already carries. The sibling ``chi`` channel
# (e-N only, no coincident sample) stayed finite throughout -- this is the
# u / f (e-e) case the 1-up/1-dn boundary uniquely exposes.
# ---------------------------------------------------------------------------


@pytest.mark.light
def test_dtn_ecp_auto_grad_e_L_r_eps_finite_single_electron_per_spin():
    """DTN ECP auto grad e L r eps finite single electron per spin."""
    from jqmc import derivatives

    trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ecp_ccpvdz_cart.h5")
    sd, _aos, _mos_up, _mos_dn, geminal, coul = read_trexio_file(trexio_file=trexio_path, store_tuple=True)

    # ACTIVE DTN(u/chi/f) with the en-cusp constraint, plus J1(exp)+J2(exp) -- the
    # exact wavefunction shape whose ECP-nonlocal r/eps autograd NaN'd.
    dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=13, N_u=2, N_chi=2, N_eN=2, N_ee=2)
    n_atom = len(sd.positions)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=sd, core_electrons=core_electrons_of(coul, n_atom), jastrow_1b_type="exp"
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=sd)
    jd = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None, jastrow_dtn_data=dtn)
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=geminal)
    ham = Hamiltonian_data(structure_data=sd, coulomb_potential_data=coul, wavefunction_data=wf)

    # 1 up + 1 dn: NO same-spin genuine pair (only the self-pair in the ratio path).
    r_up_np, r_dn_np = _draw_electrons(sd, n_up=1, n_dn=1, seed=0)
    r_up = jnp.asarray(r_up_np)
    r_dn = jnp.asarray(r_dn_np)
    RT = jnp.eye(3)

    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), "strict"
    )
    for axis, flag in (("r", "opt_r"), ("eps", "opt_eps")):
        flags = {flag: True}
        auto = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=False)[axis]
        anal = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)[axis]
        auto_leaves = jax.tree_util.tree_leaves(auto)
        anal_leaves = jax.tree_util.tree_leaves(anal)
        for leaf in auto_leaves:
            assert jnp.all(jnp.isfinite(leaf)), f"NaN/Inf in _auto d e_L/d {axis} (molecular ECP + DTN, 1up/1dn)"
        for la, lb in zip(auto_leaves, anal_leaves):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)
    jax.clear_caches()


# ---------------------------------------------------------------------------
# Incremental (rank-one) DTN grad/Laplacian advance
# ---------------------------------------------------------------------------


def _dtn_streaming_walk(dtn, r_up, r_dn, k_steps, seed, scale=0.1, moves=None):
    """Advance a DTN grad/lap streaming state over ``k_steps`` one-electron
    moves and return ``(state, r_up, r_dn)`` at the final configuration.

    ``moves`` optionally pins the sequence as ``[(is_up, idx, delta), ...]``
    (used by the cutoff- and image-crossing gates); otherwise moves are drawn
    from ``seed``. Empty spin sectors are skipped, mirroring
    ``_streaming_step_consistency_one`` in test_wave_function.py.
    """
    rng = np.random.default_rng(seed)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    r_up = np.asarray(r_up, dtype=np.float64).copy()
    r_dn = np.asarray(r_dn, dtype=np.float64).copy()
    state = _init_grads_laplacian_Jastrow_DTN_streaming_state(dtn, jnp.asarray(r_up), jnp.asarray(r_dn))
    for step in range(k_steps):
        if moves is not None:
            is_up, idx, delta = moves[step]
        else:
            is_up = bool(rng.integers(0, 2)) if (n_up and n_dn) else bool(n_up)
            idx = int(rng.integers(0, n_up if is_up else n_dn))
            delta = scale * rng.standard_normal(3)
        # COPY before mutating: `jnp.asarray` of a numpy buffer can alias it on
        # the CPU backend, so an in-place update would retroactively corrupt the
        # pre-move coordinates the state cached on the previous step.
        if is_up:
            r_up = r_up.copy()
            r_up[idx] = r_up[idx] + delta
        else:
            r_dn = r_dn.copy()
            r_dn[idx] = r_dn[idx] + delta
        state = _advance_grads_laplacian_Jastrow_DTN_streaming_state(
            dtn,
            state,
            jnp.asarray(is_up),
            jnp.asarray(idx, dtype=jnp.int32),
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
        )
    return state, r_up, r_dn


def _assert_dtn_state_matches_fresh(dtn, state, r_up, r_dn, atol, rtol, leaves=True):
    """The advanced state must equal a fresh full recompute at the SAME config,
    output-wise and (optionally) leaf-wise.

    The carried state is in the ``jastrow_grad_lap_carry`` zone (fp64 in both
    modes), so the reference is the fresh carry-zone kernel and the pair is
    that zone's strict pair."""
    # Caller and helper must agree on the already declared strict zone budget.
    expected_tolerance = get_tolerance("jastrow_grad_lap_carry", "strict")
    assert (atol, rtol) == expected_tolerance
    atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
    carry_dtype = np.dtype(get_dtype_np("jastrow_grad_lap_carry"))
    state_dtypes = {field: np.dtype(getattr(state, field).dtype) for field in ("grad_up", "grad_dn", "lap_up", "lap_dn")}
    assert set(state_dtypes.values()) == {carry_dtype}, (state_dtypes, carry_dtype)
    g_up, g_dn, l_up, l_dn = _compute_grads_and_laplacian_Jastrow_DTN_carry(dtn, jnp.asarray(r_up), jnp.asarray(r_dn))
    np.testing.assert_allclose(np.asarray(state.grad_up), np.asarray(g_up), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(state.grad_dn), np.asarray(g_dn), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(state.lap_up), np.asarray(l_up), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(state.lap_dn), np.asarray(l_dn), atol=atol, rtol=rtol)
    if leaves:
        fresh = _init_grads_laplacian_Jastrow_DTN_streaming_state(dtn, jnp.asarray(r_up), jnp.asarray(r_dn))
        expected_shapes = {
            "grad_up": (r_up.shape[0], 3),
            "grad_dn": (r_dn.shape[0], 3),
            "lap_up": (r_up.shape[0],),
            "lap_dn": (r_dn.shape[0],),
            "r_up_carts": (r_up.shape[0], 3),
            "r_dn_carts": (r_dn.shape[0], 3),
        }
        assert jax.tree_util.tree_structure(state) == jax.tree_util.tree_structure(fresh)
        assert len(jax.tree_util.tree_leaves(state)) == len(jax.tree_util.tree_leaves(fresh)) == len(expected_shapes)
        for field, shape in expected_shapes.items():
            assert np.shape(getattr(state, field)) == np.shape(getattr(fresh, field)) == shape
        np.testing.assert_allclose(np.asarray(state.grad_up), np.asarray(fresh.grad_up), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(state.grad_dn), np.asarray(fresh.grad_dn), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(state.lap_up), np.asarray(fresh.lap_up), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(state.lap_dn), np.asarray(fresh.lap_dn), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(state.r_up_carts), np.asarray(fresh.r_up_carts), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(state.r_dn_carts), np.asarray(fresh.r_dn_carts), atol=atol, rtol=rtol)


class TestDTNGradLapStreaming:
    """Regression gates for incremental DTN gradient and Laplacian updates.

    The DTN grad/Laplacian must advance INCREMENTALLY on a one-electron move
    (`O(N_atom * N_e)`) instead of being fully recomputed
    (`O(N_atom * N_e^2)`) every substep, and stay equal to the full recompute
    over the whole substep chain a branching step can produce.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True])
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
    def test_streaming_matches_fresh_k32(self, pbc, preset):
        """Gate 1: K=32 advance == fresh recompute, molecular AND PBC, every
        channel preset; state leaves compared too (Gate 5)."""
        atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
        sd = _h2_pbc_structure() if pbc else _h2_structure()
        dtn = _build_dtn(sd, preset=preset, seed=11)
        r_up, r_dn = _draw_electrons(sd, 3, 2, seed=17)
        state, r_up, r_dn = _dtn_streaming_walk(dtn, r_up, r_dn, 32, seed=23)
        _assert_dtn_state_matches_fresh(dtn, state, r_up, r_dn, atol, rtol)
        jax.clear_caches()

    @pytest.mark.parametrize(
        "k_steps",
        [
            pytest.param(32, marks=pytest.mark.light),
            pytest.param(100, marks=pytest.mark.light),
            pytest.param(1000, marks=pytest.mark.heavy),
        ],
    )
    def test_streaming_drift_accumulation(self, k_steps):
        """Gate 2: the subtractive update must not drift over the longest chain
        a branching step can produce (~220 substeps for 128 electrons; K=1000 here).

        Run under BOTH precision modes by the suite's mode matrix -- the mixed
        mode is what decides the accumulator-zone question.
        """
        atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
        sd = _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=5)
        r_up, r_dn = _draw_electrons(sd, 3, 2, seed=7)
        state, r_up, r_dn = _dtn_streaming_walk(dtn, r_up, r_dn, k_steps, seed=29, scale=0.05)
        _assert_dtn_state_matches_fresh(dtn, state, r_up, r_dn, atol, rtol, leaves=False)
        jax.clear_caches()

    @pytest.mark.light
    def test_streaming_cutoff_crossing(self):
        """Gate 3: hops that carry an electron INTO and OUT OF an atom's L_f
        shell. The subtractive update must stay exact when a pair switches
        between an identically-zero and a nonzero contribution."""
        atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
        sd = _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=3, L=1.5)
        L_f = float(np.asarray(dtn.L_f).ravel()[0])
        # electron 0 starts well inside atom 0's shell; walk it out past L_f
        # and back in, passing near the boundary.
        r_up = np.array([[0.2, 0.0, 0.0], [0.0, 0.5, 0.3]])
        r_dn = np.array([[1.4, 0.1, 0.0]])
        out = np.array([L_f * 0.9, 0.0, 0.0])
        moves = [
            (True, 0, out),  # -> ~1.1 L_f from atom 0: outside
            (True, 0, np.array([L_f * 0.6, 0.0, 0.0])),  # further out
            (True, 0, -np.array([L_f * 1.5, 0.0, 0.0])),  # back inside
            (False, 0, np.array([-1.2, 0.0, 0.0])),  # partner crosses too
            (True, 0, np.array([0.0, L_f, 0.0])),
            (False, 0, np.array([1.2, 0.0, 0.0])),
        ]
        state, r_up, r_dn = _dtn_streaming_walk(dtn, r_up, r_dn, len(moves), seed=0, moves=moves)
        _assert_dtn_state_matches_fresh(dtn, state, r_up, r_dn, atol, rtol)
        jax.clear_caches()

    @pytest.mark.light
    def test_streaming_pbc_image_crossing(self):
        """Gate 4: a move straddling the cell boundary. Every direction in the
        update must be the per-lane MINIMUM-IMAGE vector, not a raw coordinate
        difference."""
        atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
        sd = _h2_pbc_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=13)
        # NOTE the y-offset on the dn electron: with it at y=2.0 the first move
        # would put the two electrons at a minimum-image distance of EXACTLY 0
        # (|10.3 - 0.3| = 10.0 = the cell edge), i.e. coincident particles,
        # where grad r is undefined and the e-e cusp amplifies the ambiguity.
        # That is a degenerate fixture, not an image-crossing test.
        r_up = np.array([[9.7, 2.0, 2.0], [2.5, 2.0, 2.0]])
        r_dn = np.array([[0.3, 2.6, 2.0]])
        moves = [
            (True, 0, np.array([0.6, 0.0, 0.0])),  # 9.7 -> 10.3, wraps to 0.3
            (False, 0, np.array([-0.6, 0.0, 0.0])),  # 0.3 -> -0.3, wraps to 9.7
            (True, 0, np.array([0.0, 0.4, 0.0])),
        ]
        state, r_up, r_dn = _dtn_streaming_walk(dtn, r_up, r_dn, len(moves), seed=0, moves=moves)
        _assert_dtn_state_matches_fresh(dtn, state, r_up, r_dn, atol, rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("n_up,n_dn", [(3, 0), (1, 0), (4, 2), (1, 1)])
    def test_streaming_edge_cases(self, n_up, n_dn):
        """Gate 7: empty spin sectors and N_up != N_dn (the J1/J2 precedent)."""
        atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
        sd = _h2_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=2)
        r_up, r_dn = _draw_electrons(sd, n_up, n_dn, seed=31)
        state, r_up, r_dn = _dtn_streaming_walk(dtn, r_up, r_dn, 8, seed=37)
        _assert_dtn_state_matches_fresh(dtn, state, r_up, r_dn, atol, rtol)
        jax.clear_caches()

    @pytest.mark.light
    def test_streaming_walker_axis_vmap(self):
        """The init and the advance must both vmap over the walker axis: that
        is how production calls them (jqmc_gfmc.py:3464 / :2876)."""
        atol, rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
        sd = _h2_pbc_structure()
        dtn = _build_dtn(sd, preset="u_chi_f", seed=19)
        nw = 3
        r_up = np.stack([_draw_electrons(sd, 3, 2, seed=40 + w)[0] for w in range(nw)])
        r_dn = np.stack([_draw_electrons(sd, 3, 2, seed=40 + w)[1] for w in range(nw)])
        init_v = jax.vmap(lambda u, d: _init_grads_laplacian_Jastrow_DTN_streaming_state(dtn, u, d))
        adv_v = jax.vmap(
            lambda s, u, d, i: _advance_grads_laplacian_Jastrow_DTN_streaming_state(dtn, s, jnp.asarray(True), i, u, d)
        )
        state = init_v(jnp.asarray(r_up), jnp.asarray(r_dn))
        r_up_new = r_up.copy()
        idx = np.array([0, 1, 2], dtype=np.int32)
        for w in range(nw):
            r_up_new[w, idx[w]] += 0.15
        state = adv_v(state, jnp.asarray(r_up_new), jnp.asarray(r_dn), jnp.asarray(idx))
        carry_dtype = np.dtype(get_dtype_np("jastrow_grad_lap_carry"))
        state_dtypes = {field: np.dtype(getattr(state, field).dtype) for field in ("grad_up", "grad_dn", "lap_up", "lap_dn")}
        assert set(state_dtypes.values()) == {carry_dtype}, (state_dtypes, carry_dtype)
        for w in range(nw):
            walker_label = str(w)
            g_up, g_dn, l_up, l_dn = _compute_grads_and_laplacian_Jastrow_DTN_carry(
                dtn, jnp.asarray(r_up_new[w]), jnp.asarray(r_dn[w])
            )
            np.testing.assert_allclose(np.asarray(state.grad_up[w]), np.asarray(g_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_up[w]), np.asarray(l_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_dn[w]), np.asarray(g_dn), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_dn[w]), np.asarray(l_dn), atol=atol, rtol=rtol)
        jax.clear_caches()


class TestDTNElectronNucleusCuspPerSpecies:
    """The e-n cusp is a per-SPECIES property, decided by the data, not by a flag.

    A pseudized atom has no Coulomb singularity at the nucleus, so its Jastrow must be
    SMOOTH there: ``dchi/dr|_0+ = 0``. An all-electron atom keeps ``dchi/dr|_0+ =
    -Z_eff``. Imposing the all-electron condition on an ECP atom makes the Jastrow
    cancel a singularity that is not present -- measured on Cl2/ccECP it produced
    ``J = -15.9`` (``exp(J) ~ 1e-07``) with every coefficient at zero, moving the local
    energy from a physical -21.2 Ha to +10.9 Ha, and every test standing on that fixture
    was comparing implementations on a wavefunction that does not describe the system.
    The ABSENCE of the cusp under ECP is asserted rather than assumed.

    ECP records, rather than ``z_eff < Z``, are the pseudization signal. A
    mixed structure therefore exercises both arms of the same build, including
    the zero-core H-ccECP case where ``z_eff == Z``.
    """

    @staticmethod
    def _chi_derivative_at_origin(data: Jastrow_DTN_data) -> float:
        """Numerically differentiate the independent fp64 chi polynomial.

        The production builder stores ``M_beta``, ``beta_indep`` and ``b_beta``
        in fp64 in both precision modes.  Reconstructing those coefficients in
        NumPy gives an evaluator independent of the XLA DTN kernel.  A fourth-
        order forward stencil then checks the actual polynomial slope to the
        ``1e-8`` cusp contract without making a float32 cancellation test the
        limiting oracle.

        ``_build`` translates the probed atom to the origin. Other atoms are
        farther away than ``L_chi`` and therefore contribute exactly zero.
        """
        atom_index = int(np.argmin(np.linalg.norm(np.asarray(data.structure_data.positions), axis=1)))
        species = int(np.asarray(data.species_index)[atom_index])
        beta_full = np.einsum(
            "Iki,Ii->Ik",
            np.asarray(data.M_beta, dtype=np.float64),
            np.asarray(data.beta_indep, dtype=np.float64),
        ) + np.asarray(data.b_beta, dtype=np.float64)
        coefficients = beta_full[species]
        cutoff = float(data.L_chi[species])
        smoothness = int(data.cutoff_smoothness)

        def chi_at(radius: float) -> float:
            polynomial = np.polynomial.polynomial.polyval(radius, coefficients)
            return float((cutoff - radius) ** smoothness * polynomial)

        step = 1.0e-3
        return (
            -25.0 * chi_at(0.0)
            + 48.0 * chi_at(step)
            - 36.0 * chi_at(2.0 * step)
            + 16.0 * chi_at(3.0 * step)
            - 3.0 * chi_at(4.0 * step)
        ) / (12.0 * step)

    @staticmethod
    def _build(
        atomic_numbers,
        symbols,
        labels,
        positions,
        z_eff,
        pseudized,
        species_index=None,
        measured_atom=0,
    ):
        """Build the DTN data with ``measured_atom`` translated onto the origin."""
        pos = np.asarray(positions, dtype=np.float64)
        pos = pos - pos[measured_atom]
        sd = Structure_data.build(
            pbc_flag=False,
            positions=pos,
            atomic_numbers=tuple(atomic_numbers),
            element_symbols=tuple(symbols),
            atomic_labels=tuple(labels),
        )
        return sd, Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=tuple(z_eff),
            pseudized_per_species=tuple(pseudized),
            species_index=species_index,
            use_u=False,
            use_chi=True,
            use_f=False,
            cutoff_smoothness=4,
            n_order_chi=4,
            L_chi=3.0,
            L_u=3.0,
            L_f=3.0,
            random_init=False,
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "atomic_number,symbol,z_eff,pseudized",
        [(3, "Li", 3.0, False), (17, "Cl", 7.0, True), (1, "H", 1.0, True)],
        ids=["AE", "ECP", "zero-core-ECP"],
    )
    def test_physical_cusp_target_keeps_constraints_enabled(self, atomic_number, symbol, z_eff, pseudized):
        """AE slope -Z_eff and ECP slope zero both retain the chi constraint.

        Nonzero independent coefficients rule out an identically zero ECP
        polynomial. The derivative at the origin is computed algebraically
        from the stored polynomial, with no FD or cross-precision reference.
        """
        _sd, data = self._build((atomic_number,), (symbol,), (symbol + "1",), [[0.0, 0.0, 0.0]], (z_eff,), (pseudized,))
        rng = np.random.default_rng(20260920)
        data = data.replace(beta_indep=rng.normal(size=np.asarray(data.beta_indep).shape))
        data.sanity_check()
        assert data.apply_en_cusp is True
        coefficients = np.einsum("Iki,Ii->Ik", np.asarray(data.M_beta), np.asarray(data.beta_indep)) + np.asarray(data.b_beta)
        assert np.any(coefficients != 0)
        cutoff = np.asarray(data.L_chi)
        power = data.cutoff_smoothness
        slope = cutoff**power * coefficients[:, 1] - power * cutoff ** (power - 1) * coefficients[:, 0]
        expected = np.array([0.0 if pseudized else -z_eff])
        np.testing.assert_allclose(slope, expected, atol=ATOL_DTN_EN_CUSP_CONSTRAINT, rtol=RTOL_DTN_EN_CUSP_CONSTRAINT)

    @pytest.mark.light
    @pytest.mark.parametrize(
        "atomic_number,symbol,z_eff,pseudized",
        [(3, "Li", 3.0, False), (17, "Cl", 7.0, True), (1, "H", 1.0, True)],
        ids=["AE", "ECP", "zero-core-ECP"],
    )
    def test_cusp_constraints_cannot_be_disabled(self, atomic_number, symbol, z_eff, pseudized):
        """ECP zero cusp never authorizes an unconstrained chi coefficient."""
        _sd, data = self._build((atomic_number,), (symbol,), (symbol + "1",), [[0.0, 0.0, 0.0]], (z_eff,), (pseudized,))
        assert data.apply_en_cusp is True
        unsafe = data.replace(apply_en_cusp=False)
        with pytest.raises(ValueError, match=r"apply_en_cusp=False.*unsafe"):
            unsafe.sanity_check()

    @pytest.mark.light
    def test_all_electron_species_keeps_its_cusp(self):
        """AE only: no ECP record, so the nucleus keeps the ``-Z_eff`` cusp."""
        _sd, data = self._build((3,), ("Li",), ("Li1",), [[0.0, 0.0, 0.0]], (3.0,), (False,))
        deriv = self._chi_derivative_at_origin(data)
        np.testing.assert_allclose(
            deriv,
            -3.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )

    @pytest.mark.light
    def test_pseudized_species_is_cuspless(self):
        """Both ordinary and zero-core ECP records make chi smooth."""
        _sd, data = self._build((17,), ("Cl",), ("Cl1",), [[0.0, 0.0, 0.0]], (7.0,), (True,))
        deriv = self._chi_derivative_at_origin(data)
        np.testing.assert_allclose(
            deriv,
            0.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )
        # Non-vacuity: the all-electron arm of the SAME build is a different number, so
        # this is not passing because chi is identically flat.
        _sd_ae, data_ae = self._build((7,), ("N",), ("N1",), [[0.0, 0.0, 0.0]], (7.0,), (False,))
        np.testing.assert_allclose(
            self._chi_derivative_at_origin(data_ae),
            -7.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )

        from jqmc.qe_to_jqmc import build_jastrow_dtn_from_options

        fixture = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files", "H_ecp_ccpvqz.h5")
        fixture = os.path.abspath(fixture)
        sd, _aos, _mos_up, _mos_dn, _geminal, coulomb = read_trexio_file(fixture, store_tuple=True)
        np.testing.assert_array_equal(coulomb.pseudized_per_atom, np.ones(sd.natom, dtype=np.bool_))
        np.testing.assert_allclose(
            coulomb._effective_charges,
            np.asarray(sd.atomic_numbers, dtype=np.float64),
            atol=ATOL_BIT_EXACT,
            rtol=RTOL_BIT_EXACT,
        )
        h_ecp_data = build_jastrow_dtn_from_options(
            sd,
            jastrow_dtn="u_chi",
            coulomb_potential_data=coulomb,
        )
        np.testing.assert_allclose(
            self._chi_derivative_at_origin(h_ecp_data),
            0.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )

    @pytest.mark.light
    def test_mixed_ae_and_ecp_decides_per_species(self):
        """Mixed: one AE nucleus and one pseudized nucleus in the SAME structure.

        Species are ordered by atomic number, so H (Z=1, all-electron) keeps its cusp
        while Cl (Z=17, z_eff=7) is smooth. A global flag cannot express this.
        """
        positions = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        sd = Structure_data.build(
            pbc_flag=False,
            positions=np.asarray(positions),
            atomic_numbers=(1, 17),
            element_symbols=("H", "Cl"),
            atomic_labels=("H1", "Cl1"),
        )
        coulomb = Coulomb_potential_data(
            structure_data=sd,
            ecp_flag=True,
            z_cores=(0.0, 10.0),
            # l_max = 0 (a single non-local s channel), so the local tag is
            # l_max + 1 = 1. This test only exercises `pseudized_per_atom`, so
            # the physical content of both terms is a no-op (coefficient 0.0,
            # legitimate per Coulomb_potential_data.sanity_check -- that is
            # exactly how PySCF marks H/He placeholders); what matters is that
            # every l the tag declares actually has a term, since
            # sanity_check rejects a declared-but-absent l.
            max_ang_mom_plus_1=(0, 1),
            num_ecps=2,
            ang_moms=(1, 0),
            nucleus_index=(1, 1),
            exponents=(1.0, 1.0),
            coefficients=(0.0, 0.0),
            powers=(2, 0),
        )
        coulomb.sanity_check()
        np.testing.assert_array_equal(coulomb.pseudized_per_atom, (False, True))
        args = (
            (1, 17),
            ("H", "Cl"),
            ("H1", "Cl1"),
            positions,
            tuple(coulomb._effective_charges),
            tuple(coulomb.pseudized_per_atom),
        )
        ae = self._chi_derivative_at_origin(self._build(*args, measured_atom=0)[1])
        ecp = self._chi_derivative_at_origin(self._build(*args, measured_atom=1)[1])
        np.testing.assert_allclose(
            ae,
            -1.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )
        np.testing.assert_allclose(
            ecp,
            0.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )
        assert abs(ae - ecp) > 0.5, "the two species must be decided differently, not both flat"

    @pytest.mark.light
    def test_explicit_species_index_binds_policy_to_the_right_species(self):
        """An explicit species table keeps each element's ECP policy aligned."""
        positions = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0]]
        args = (
            (17, 1, 1),
            ("Cl", "H", "H"),
            ("Cl1", "H1", "H2"),
            positions,
            (7.0, 1.0, 1.0),
            (True, False, False),
        )

        def probe(atom_index: int) -> float:
            data = self._build(*args, species_index=(0, 1, 2), measured_atom=atom_index)[1]
            return self._chi_derivative_at_origin(data)

        ecp, ae_1, ae_2 = probe(0), probe(1), probe(2)
        np.testing.assert_allclose(
            ecp,
            0.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )
        np.testing.assert_allclose(
            ae_1,
            -1.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )
        np.testing.assert_allclose(
            ae_2,
            -1.0,
            atol=ATOL_DTN_EN_CUSP_CONSTRAINT,
            rtol=RTOL_DTN_EN_CUSP_CONSTRAINT,
        )

    @pytest.mark.light
    def test_a_species_mixing_elements_is_refused(self):
        """One species must be one element, or its bare Z -- and its cusp -- is ambiguous.

        Silently picking one member's Z would impose that atom's cusp on the other
        atom's nucleus, which is exactly the class of defect these tests exist to stop.
        """
        positions = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        with pytest.raises(ValueError, match="mixes atomic numbers"):
            self._build(
                (1, 17),
                ("H", "Cl"),
                ("H1", "Cl1"),
                positions,
                (1.0,),
                (False,),
                species_index=(0, 0),
            )


# ---------------------------------------------------------------------------
# Comoving strain through ``Jastrow_data._apply_strain`` (``cutoff_scale`` leaf)
# ---------------------------------------------------------------------------


class TestDTNComovingApplyStrain:
    """``_apply_strain`` follows the comoving cutoff policy for periodic DTN factors.

    ``Jastrow_DTN_data.cutoff_scale`` (pytree leaf, ``1.0`` at rest) records ``det(F)^(1/3)``;
    the kernels use ``L * cutoff_scale`` and the inflated coefficients ride the minimum-norm
    constraint tangent to first order.  Autograd / finite differences taken THROUGH
    ``_apply_strain`` therefore reproduce the analytic stress kernels, which add the same
    policy term (``dJ/dln(ell_V) / 3 * I``) in closed form.
    """

    @staticmethod
    def _pbc_dtn(seed: int = 7) -> Jastrow_DTN_data:
        return Jastrow_DTN_data.build(
            structure_data=_h2_pbc_structure(),
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
            cutoff_ratio_u=0.8,
            cutoff_ratio_chi=0.7,
            cutoff_ratio_f=0.9,
            random_init=True,
            seed=seed,
        )

    @staticmethod
    def _wrap(dtn: Jastrow_DTN_data) -> Jastrow_data:
        return Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
        )

    @pytest.mark.light
    def test_rest_state_is_bitwise_plain(self):
        """``cutoff_scale == 1.0`` reproduces the plain cutoffs and inflation bit for bit."""
        from jqmc.jastrow_factor import _dtn_effective_cutoffs, _dtn_inflate_alpha, _dtn_inflate_beta, _dtn_inflate_gamma

        dtn = self._pbc_dtn()
        assert float(dtn.cutoff_scale) == 1.0
        L_u, L_chi, L_f = _dtn_effective_cutoffs(dtn, jnp.float64)
        assert float(L_u) == float(dtn.L_u)
        np.testing.assert_array_equal(np.asarray(L_chi), np.asarray(dtn.L_chi, dtype=np.float64))
        np.testing.assert_array_equal(np.asarray(L_f), np.asarray(dtn.L_f, dtype=np.float64))
        # Same contraction (jnp.einsum, same operand order) as the inflation itself, so the only
        # thing being tested is that the transport adds EXACTLY zero at rest.
        f64 = jnp.float64
        plain_alpha = jnp.einsum("ski,si->sk", jnp.asarray(dtn.M_alpha, f64), jnp.asarray(dtn.alpha_indep, f64)) + jnp.asarray(
            dtn.b_alpha, f64
        )
        np.testing.assert_array_equal(np.asarray(_dtn_inflate_alpha(dtn, f64)), np.asarray(plain_alpha))
        plain_beta = jnp.einsum("Iki,Ii->Ik", jnp.asarray(dtn.M_beta, f64), jnp.asarray(dtn.beta_indep, f64)) + jnp.asarray(
            dtn.b_beta, f64
        )
        np.testing.assert_array_equal(np.asarray(_dtn_inflate_beta(dtn, f64)), np.asarray(plain_beta))
        plain_gamma = jnp.einsum("Ilmni,Ii->Ilmn", jnp.asarray(dtn.M_gamma, f64), jnp.asarray(dtn.gamma_indep, f64))
        np.testing.assert_array_equal(np.asarray(_dtn_inflate_gamma(dtn, f64)), np.asarray(plain_gamma))

    @pytest.mark.light
    def test_apply_strain_scales_cutoffs_and_transports_coefficients(self):
        """``F = s I``: cutoffs scale by ``s``; the transported full coefficients satisfy the cusp
        constraint at the NEW cutoff to second order in ``ln s`` (first order in the tangent)."""
        from jqmc.jastrow_factor import _dtn_effective_cutoffs, _dtn_inflate_alpha, _dtn_inflate_beta

        dtn = self._pbc_dtn()
        C = int(dtn.cutoff_smoothness)
        for s in (1.0 + 1.0e-3, 1.0 - 1.0e-3):
            strained = self._wrap(dtn)._apply_strain(s * jnp.eye(3, dtype=jnp.float64)).jastrow_dtn_data
            assert abs(float(strained.cutoff_scale) - s) < ATOL_DTN_COMOVING_CUTOFF_SCALE
            L_u, L_chi, L_f = _dtn_effective_cutoffs(strained, jnp.float64)
            assert abs(float(L_u) - s * float(dtn.L_u)) < ATOL_DTN_COMOVING_CUTOFF_SCALE
            np.testing.assert_allclose(
                np.asarray(L_chi),
                s * np.asarray(dtn.L_chi, dtype=np.float64),
                atol=ATOL_BIT_EXACT,
                rtol=RTOL_DTN_COMOVING_CUTOFF_SCALE,
            )
            np.testing.assert_allclose(
                np.asarray(L_f),
                s * np.asarray(dtn.L_f, dtype=np.float64),
                atol=ATOL_BIT_EXACT,
                rtol=RTOL_DTN_COMOVING_CUTOFF_SCALE,
            )
            # e-e cusp ((L - r)^C form): C alpha_0 - L alpha_1 = -cusp / L^(C-1)  (cusp = 1/4 parallel, 1/2 antiparallel)
            alpha = np.asarray(_dtn_inflate_alpha(strained, jnp.float64))
            L = float(L_u)
            for spin, cusp in enumerate((0.25, 0.5)):
                residual = C * alpha[spin, 0] - L * alpha[spin, 1] - (-cusp / L ** (C - 1))
                assert abs(residual) < ATOL_DTN_COMOVING_CUSP_TRANSPORT, (spin, residual)
            # e-N cusp per species (Z_eff = 1, not pseudized): C beta_0 - L beta_1 = Z / L^(C-1)
            beta = np.asarray(_dtn_inflate_beta(strained, jnp.float64))
            for I in range(beta.shape[0]):
                L = float(np.asarray(L_chi)[I])
                residual = C * beta[I, 0] - L * beta[I, 1] - 1.0 / L ** (C - 1)
                assert abs(residual) < ATOL_DTN_COMOVING_CUSP_TRANSPORT, (I, residual)
        # Without the transport the same cusp residual is first order in ln s (sanity of the check itself).
        untransported = dtn.replace(cutoff_scale=1.0)
        alpha0 = np.asarray(_dtn_inflate_alpha(untransported, jnp.float64))
        L = float(dtn.L_u) * (1.0 + 1.0e-3)
        residual0 = C * alpha0[0, 0] - L * alpha0[0, 1] - (-0.25 / L ** (C - 1))
        # tol-ok: non-vacuity floor -- without the transport the cusp residual must be
        # FIRST order in ln s (~1e-3 * O(1)), i.e. far above the transported bound above.
        assert abs(residual0) > 1.0e-5

    @pytest.mark.light
    def test_strain_jacrev_equals_fixed_cutoff_plus_policy(self):
        """``jacrev`` of ``J_DTN`` through ``_apply_strain`` (comoving) equals the fixed-cutoff
        ``jacrev`` plus the analytic policy term ``dJ/dln(ell_V) / 3 * I``.  Full precision uses
        the fp64 identity tolerance; mixed compares the two results at the crossed fp32
        ``jastrow_eval`` boundary and permits at most one representable-value step."""
        from jqmc.derivatives.dtn import dtn_cutoff_policy_value

        dtn = self._pbc_dtn()
        jd = self._wrap(dtn)
        rng = np.random.default_rng(0)
        r_up = jnp.asarray(rng.uniform(0.0, 10.0, size=(3, 3)))
        r_dn = jnp.asarray(rng.uniform(0.0, 10.0, size=(3, 3)))

        def J_comoving(eps):
            F = jnp.eye(3, dtype=jnp.float64) + eps
            return compute_Jastrow_DTN(jd._apply_strain(F).jastrow_dtn_data, r_up @ F.T, r_dn @ F.T)

        def J_fixed(eps):
            F = jnp.eye(3, dtype=jnp.float64) + eps
            d = dtn.replace(structure_data=_strain_structure_data(dtn.structure_data, F))
            return compute_Jastrow_DTN(d, r_up @ F.T, r_dn @ F.T)

        eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
        g_comoving = np.asarray(jax.jacrev(J_comoving)(eps0))
        g_fixed = np.asarray(jax.jacrev(J_fixed)(eps0))
        policy = float(dtn_cutoff_policy_value(dtn, r_up, r_dn))
        # tol-ok: non-vacuity floor, not a tolerance -- the fixture must exercise a
        # policy term big enough for the identity below to have content.
        assert abs(policy) > 1.0e-3
        expected = g_fixed + policy / 3.0 * np.eye(3)
        if get_dtype_np("jastrow_eval") == np.dtype(np.float32):
            np.testing.assert_array_max_ulp(
                g_comoving.astype(np.float32),
                expected.astype(np.float32),
                maxulp=MAXULP_DTN_COMOVING_POLICY_IDENTITY_MIXED,
            )
        else:
            np.testing.assert_allclose(
                g_comoving,
                expected,
                rtol=RTOL_DTN_COMOVING_POLICY_IDENTITY,
                atol=ATOL_DTN_COMOVING_POLICY_IDENTITY,
            )

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="central finite difference of the fp32 Jastrow value is round-off dominated",
        alternative_guarantee_ref="MATH-JAS-COMOVING-STRAIN-001",
    )
    def test_strain_jacrev_matches_isotropic_finite_difference(self):
        """The comoving ``J_DTN`` strain ``jacrev`` trace matches a central finite difference
        along isotropic strain in full precision."""

        dtn = self._pbc_dtn()
        jd = self._wrap(dtn)
        rng = np.random.default_rng(0)
        r_up = jnp.asarray(rng.uniform(0.0, 10.0, size=(3, 3)))
        r_dn = jnp.asarray(rng.uniform(0.0, 10.0, size=(3, 3)))

        def J_comoving(eps):
            F = jnp.eye(3, dtype=jnp.float64) + eps
            return compute_Jastrow_DTN(jd._apply_strain(F).jastrow_dtn_data, r_up @ F.T, r_dn @ F.T)

        eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
        g_comoving = np.asarray(jax.jacrev(J_comoving)(eps0))
        h = 1.0e-5
        fd_trace = (float(J_comoving(h * jnp.eye(3))) - float(J_comoving(-h * jnp.eye(3)))) / (2.0 * h)
        assert abs(fd_trace - np.trace(g_comoving)) < ATOL_DTN_COMOVING_POLICY_FD

    @pytest.mark.light
    def test_molecular_apply_strain_keeps_cutoff_scale(self):
        """Molecular DTN cutoffs are absolute: ``_apply_strain`` leaves ``cutoff_scale`` at 1."""
        dtn = Jastrow_DTN_data.build(structure_data=_h2_structure(), z_eff_per_species=(1.0,), pseudized_per_species=(False,))
        strained = self._wrap(dtn)._apply_strain(1.01 * jnp.eye(3, dtype=jnp.float64)).jastrow_dtn_data
        assert float(strained.cutoff_scale) == 1.0

    @pytest.mark.light
    def test_sanity_check_rejects_bad_cutoff_scale(self):
        """DTN sanity checking rejects zero and negative comoving cutoff scales."""
        dtn = self._pbc_dtn()
        with pytest.raises(ValueError, match="cutoff_scale"):
            dtn.replace(cutoff_scale=0.0).sanity_check()
        with pytest.raises(ValueError, match="cutoff_scale"):
            dtn.replace(cutoff_scale=-1.0).sanity_check()


# ---------------------------------------------------------------------------
# Issue #68: periodic DTN defaults use the true Wigner-Seitz inscribed radius,
# and the minimum image is exact below it on oblique cells.
# ---------------------------------------------------------------------------
#: Lattices (standard basis) and the basis changes of the DTN basis-invariance check.
#: ``non_reduced_transform`` is far from reduced, so the kernels wrap in the LLL-reduced
#: basis (``mic_basis_change`` != identity); the other bases wrap in the given basis.
DTN_WS_BASES = ("identity", "single_shear", "non_reduced_transform")
DTN_WS_LATTICE_CASES = (
    ("fcc_a8", DTN_WS_BASES),
    ("bcc_a8", DTN_WS_BASES),
    ("hexagonal_a6_c10", DTN_WS_BASES),
    ("orthorhombic_6_7_8", DTN_WS_BASES),
)
#: Fractional atom positions (standard basis) and electron counts of the fixture.
DTN_WS_ATOM_FRACTIONS = np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.1]])
DTN_WS_N_UP, DTN_WS_N_DN = 6, 5


def _brute_force_minimum_distance(displacements: np.ndarray, reduced_cell: np.ndarray, n_box: int = 3) -> np.ndarray:
    """Exact minimum-image length of ``displacements`` (``(..., 3)``), independent of production code.

    The inputs are first reduced into the home parallelepiped of ``reduced_cell`` (a
    standard, reduced basis); every minimum image is then within ``|n_k| <= n_box``
    of that representative for the cells used here (``n_box`` = 3 covers the
    covering-radius bound ``|n_k| <= rho |g_k| + 1/2`` with ``rho <= sqrt(sum |b_i|^2) / 2``).
    """
    reduced_cell = np.asarray(reduced_cell, dtype=np.float64)
    inv = np.linalg.inv(reduced_cell)
    rho = 0.5 * np.sqrt(np.sum(reduced_cell * reduced_cell))
    assert np.all(np.ceil(rho * np.linalg.norm(inv, axis=0) + 0.5) <= n_box)
    frac = displacements @ inv
    home = (frac - np.round(frac)) @ reduced_cell
    grid = np.array(list(itertools.product(range(-n_box, n_box + 1), repeat=3)), dtype=np.float64) @ reduced_cell
    return np.min(np.linalg.norm(home[..., None, :] + grid, axis=-1), axis=-1)


def _ws_basis_fixture(lattice_name: str, basis_name: str) -> tuple:
    """Default periodic DTN (random coefficients) on ``U @ L`` plus a fixed electron configuration.

    The atoms and electrons have the same Cartesian coordinates for every basis of the
    same lattice, so every physical quantity must agree between bases.
    """
    from tests.test_jqmc_kernel.test_structure import (
        WS_RADIUS_BASIS_CHANGES,
        WS_RADIUS_LATTICES,
        brute_force_half_shortest_lattice_vector,
    )

    lattice = WS_RADIUS_LATTICES[lattice_name]
    cell = WS_RADIUS_BASIS_CHANGES[basis_name] @ lattice
    atoms = DTN_WS_ATOM_FRACTIONS @ lattice
    structure = make_pbc_structure(
        cell,
        positions=atoms,
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )
    dtn = Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=(1.0,),
        pseudized_per_species=(False,),
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        random_init=True,
        seed=68,
    )
    rng = np.random.default_rng(seed=20261012)
    electrons = rng.uniform(0.0, 1.0, size=(DTN_WS_N_UP + DTN_WS_N_DN, 3)) @ lattice
    # Up electron 0 sits next to atom 0, and up electron 1 next to up electron 0,
    # at separations inside the cutoffs that the plain parallelepiped wrap in the
    # sheared basis maps to a longer image (found by a seeded search).
    sheared = WS_RADIUS_BASIS_CHANGES["single_shear"] @ lattice
    inv_sheared = np.linalg.inv(sheared)
    cutoff = DTN_DEFAULT_CUTOFF_RATIO * brute_force_half_shortest_lattice_vector(lattice)
    found = []
    for _ in range(100000):
        direction = rng.normal(size=3)
        candidate = direction / np.linalg.norm(direction) * cutoff * rng.uniform(0.5, 0.99)
        frac = candidate @ inv_sheared
        if np.linalg.norm((frac - np.round(frac)) @ sheared) > cutoff:
            found.append(candidate)
        if len(found) == 2:
            break
    assert len(found) == 2, lattice_name
    electrons[0] = atoms[0] + found[0]
    electrons[1] = electrons[0] + found[1]
    return lattice, cell, atoms, dtn, electrons[:DTN_WS_N_UP], electrons[DTN_WS_N_UP:]


def _round_only_misses(cell: np.ndarray, displacements: np.ndarray, exact: np.ndarray, cutoff: float) -> int:
    """Count pairs inside ``cutoff`` whose plain parallelepiped wrap in ``cell`` is not the minimum image."""
    frac = displacements @ np.linalg.inv(cell)
    wrapped = np.linalg.norm((frac - np.round(frac)) @ cell, axis=-1)
    return int(np.sum((exact < cutoff) & (wrapped > cutoff)))


@pytest.mark.light
def test_periodic_dtn_default_cutoffs_and_value_are_basis_independent_on_oblique_lattices():
    """Default periodic DTN cutoffs are 0.95 x the WS inradius, and J, grad, lap do not depend on the basis.

    For FCC, BCC and hexagonal primitive cells and an orthorhombic lattice, each
    written in three bases ``U @ L`` (``det U = 1``; identity, a single shear and the
    far-from-reduced transform, which takes the LLL wrap-basis path):

    * ``L_u = L_chi = 0.95 R`` and ``L_f = 0.95 R / 2`` with ``R`` the brute-force
      half shortest lattice vector (independent reference);
    * ``_dtn_distance_rows`` and the derivative-module ``_md`` give the brute-force
      minimum-image length for every e-e and e-N pair shorter than ``R``;
    * ``compute_Jastrow_DTN`` and ``compute_grads_and_laplacian_Jastrow_DTN`` agree
      between the bases of the same lattice at the same Cartesian configuration.

    Non-vacuity: for every lattice some pair inside a cutoff is wrapped to a longer
    image by the plain parallelepiped wrap in one of the bases, so the check fails if
    only the radius or only the image search were fixed.
    """
    from tests.test_jqmc_kernel.test_structure import brute_force_half_shortest_lattice_vector

    atol64, rtol64 = ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
    atol_v, rtol_v = get_tolerance("jastrow_eval", "strict")
    atol_g, rtol_g = get_tolerance("jastrow_grad_lap", "strict")
    for lattice_name, basis_names in DTN_WS_LATTICE_CASES:
        results = {}
        misses = 0
        for basis_name in basis_names:
            lattice, cell, atoms, dtn, r_up, r_dn = _ws_basis_fixture(lattice_name, basis_name)
            label = f"{lattice_name}/{basis_name}"
            radius = brute_force_half_shortest_lattice_vector(lattice)
            np.testing.assert_allclose(
                float(dtn.L_u), DTN_DEFAULT_CUTOFF_RATIO * radius, atol=atol64, rtol=rtol64, err_msg=label
            )
            np.testing.assert_allclose(
                np.asarray(dtn.L_chi, dtype=np.float64),
                DTN_DEFAULT_CUTOFF_RATIO * radius,
                atol=atol64,
                rtol=rtol64,
                err_msg=label,
            )
            np.testing.assert_allclose(
                np.asarray(dtn.L_f, dtype=np.float64),
                DTN_DEFAULT_CUTOFF_RATIO * 0.5 * radius,
                atol=atol64,
                rtol=rtol64,
                err_msg=label,
            )
            # Non-vacuity of the reduced-wrap path: the far-from-reduced basis must take
            # it, and the standard basis must keep its own wrap.
            takes_reduced_wrap = tuple(map(tuple, dtn.mic_basis_change)) != ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            if basis_name in ("identity", "non_reduced_transform"):
                assert takes_reduced_wrap == (basis_name == "non_reduced_transform"), (label, dtn.mic_basis_change)

            electrons = np.concatenate((r_up, r_dn), axis=0)
            for partner_label, partners in (("ee", electrons), ("eN", atoms)):
                displacements = electrons[:, None, :] - partners[None, :, :]
                exact = _brute_force_minimum_distance(displacements, lattice)
                rows = np.asarray(_dtn_distance_rows(dtn, jnp.asarray(electrons), jnp.asarray(partners), jnp.float64))
                md = np.linalg.norm(np.asarray(_md(jnp.asarray(displacements), dtn)), axis=-1)
                inside = exact < radius
                np.testing.assert_allclose(
                    rows[inside], exact[inside], atol=atol64, rtol=rtol64, err_msg=f"{label}/{partner_label}"
                )
                np.testing.assert_allclose(
                    md[inside], exact[inside], atol=atol64, rtol=rtol64, err_msg=f"{label}/{partner_label}/_md"
                )
                cutoff = float(dtn.L_u) if partner_label == "ee" else float(np.max(np.asarray(dtn.L_chi)))
                misses += _round_only_misses(cell, displacements, exact, cutoff)

            value = float(compute_Jastrow_DTN(dtn, jnp.asarray(r_up), jnp.asarray(r_dn)))
            grads_laps = [
                np.asarray(x, dtype=np.float64)
                for x in compute_grads_and_laplacian_Jastrow_DTN(dtn, jnp.asarray(r_up), jnp.asarray(r_dn))
            ]
            assert np.isfinite(value) and all(np.all(np.isfinite(x)) for x in grads_laps), label
            results[basis_name] = (value, grads_laps)
        assert misses > 0, f"{lattice_name}: no pair inside a cutoff needs the image shell"
        value_a, grads_a = results[basis_names[0]]
        for basis_name in basis_names[1:]:
            value_b, grads_b = results[basis_name]
            label = f"{lattice_name}/{basis_name} vs {basis_names[0]}"
            np.testing.assert_allclose(value_b, value_a, atol=atol_v, rtol=rtol_v, err_msg=f"{label}: J")
            for name, got, ref in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_b, grads_a):
                np.testing.assert_allclose(got, ref, atol=atol_g, rtol=rtol_g, err_msg=f"{label}: {name}")
        jax.clear_caches()


@pytest.mark.light
def test_periodic_dtn_minimum_image_plan_is_overflow_safe_on_extreme_unreduced_basis(monkeypatch):
    """An extreme unreduced basis selects the reduced zero-shell DTN plan and evaluates without a huge shell.

    ``L = diag(6, 7, 8)`` Bohr written as ``U @ L`` with the unimodular
    ``U = [[1, 3e6, 3e6], [0, 1, 3e6], [0, 0, 1]]``. The given-basis shell has half-widths
    ``(1, 2000000, 5999998000000)``, about ``1.44e20`` candidates, whose cardinality
    overflows a 64-bit product; the LLL-reduced basis of the same lattice is a signed
    permutation of the rectangular rows and needs no shell. Checks:

    * the static plan takes an integer unimodular ``M`` whose rows ``M @ cell`` are the
      rectangular rows (exact), with shell ``(0, 0, 0)``, and ``Jastrow_DTN_data.build``
      stores that plan;
    * public ``compute_Jastrow_DTN`` (u-only, default cutoffs, default initial coefficients)
      finishes and equals the elementary cusp polynomial ``sum -(a_s L_u / C) (1 - r / L_u)^C``
      with ``a = 1/4`` (parallel) and ``1/2`` (antiparallel) over rectangular minimum-image
      distances (independent reference);
    * no shell larger than the LLL bound of 27 candidates is ever expanded (allocation
      sentinel on the shift expansion, so the unfixed code fails without allocating);
    * an infeasible shell is rejected with ``ValueError``: by the shift expansion just
      above ``MINIMUM_IMAGE_SHELL_MAX_CANDIDATES`` and by ``sanity_check`` for the
      identity-basis plan with the huge shell.
    """
    import jqmc.jastrow_factor as jastrow
    from jqmc.structure import _minimum_image_reduction_plan, _minimum_image_shell_shifts

    rectangle = np.array([6.0, 7.0, 8.0])
    unimodular = np.array([[1, 3000000, 3000000], [0, 1, 3000000], [0, 0, 1]], dtype=np.float64)
    cell = unimodular @ np.diag(rectangle)
    lll_bound = 27

    expanded = []
    original_shifts = jastrow._minimum_image_shell_shifts

    def bounded_shifts(half_widths):
        size = 1
        for h in half_widths:
            size *= 2 * int(h) + 1
        expanded.append(tuple(int(h) for h in half_widths))
        if size > lll_bound:
            pytest.fail(f"DTN minimum-image shell {half_widths} with {size} candidates was expanded")
        return original_shifts(half_widths)

    monkeypatch.setattr(jastrow, "_minimum_image_shell_shifts", bounded_shifts)

    basis_change, half_widths = _minimum_image_reduction_plan(cell)
    change = np.asarray(basis_change, dtype=np.float64)
    assert half_widths == (0, 0, 0), (basis_change, half_widths)
    assert abs(round(float(np.linalg.det(change)))) == 1, basis_change
    reduced_rows = change @ cell
    assert np.count_nonzero(reduced_rows) == 3, reduced_rows
    assert sorted(np.abs(reduced_rows).sum(axis=1).tolist()) == sorted(rectangle.tolist()), reduced_rows
    assert sorted(np.argmax(np.abs(reduced_rows), axis=1).tolist()) == [0, 1, 2], reduced_rows

    structure = make_pbc_structure(cell)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=(1.0,),
        pseudized_per_species=(False,),
        use_chi=False,
        use_f=False,
    )
    assert tuple(map(tuple, dtn.mic_basis_change)) == basis_change
    assert tuple(dtn.mic_shell_half_widths) == (0, 0, 0)

    # Every pair is inside L_u = 0.95 * 3 Bohr, and each minimum image needs wrapping.
    r_up = np.array([[1.0, 1.0, 1.0], [6.0, -5.2, 8.3]])
    r_dn = np.array([[6.5, 1.4, -6.7]])
    L_u, C = float(dtn.L_u), int(dtn.cutoff_smoothness)

    def rectangular_distance(a, b):
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return float(np.linalg.norm(d - rectangle * np.round(d / rectangle)))

    pairs = [(rectangular_distance(r_up[0], r_up[1]), 0.25)]
    pairs += [(rectangular_distance(r, r_dn[0]), 0.5) for r in r_up]
    assert all(r < L_u for r, _ in pairs), pairs
    reference = sum(-(a * L_u / C) * (1.0 - r / L_u) ** C for r, a in pairs)

    value = float(compute_Jastrow_DTN(dtn, jnp.asarray(r_up), jnp.asarray(r_dn)))
    atol_v, rtol_v = get_tolerance("jastrow_eval", "strict")
    np.testing.assert_allclose(value, reference, atol=atol_v, rtol=rtol_v)
    assert expanded and all(h == (0, 0, 0) for h in expanded), expanded

    # Guards: a shell one candidate above the limit (cheap to build if unguarded) and the
    # identity-basis plan with the huge shell are rejected before any expansion.
    from jqmc._setting import MINIMUM_IMAGE_SHELL_MAX_CANDIDATES

    over_limit = (0, 0, MINIMUM_IMAGE_SHELL_MAX_CANDIDATES // 2)
    with pytest.raises(ValueError, match="MINIMUM_IMAGE_SHELL_MAX_CANDIDATES"):
        _minimum_image_shell_shifts(over_limit)
    huge = dtn.replace(mic_basis_change=((1, 0, 0), (0, 1, 0), (0, 0, 1)), mic_shell_half_widths=(1, 2000000, 5999998000000))
    with pytest.raises(ValueError, match="MINIMUM_IMAGE_SHELL_MAX_CANDIDATES"):
        huge.sanity_check()
    jax.clear_caches()

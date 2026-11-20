"""collections of unit tests."""

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

import itertools
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_jqmc_kernel._gdd_fd_adapters import fd_case_parameters, record_fd_case

pytestmark = [pytest.mark.with_chdir]
from numpy import linalg as LA
from numpy.testing import assert_allclose

# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# All AO production / oracle helpers live in a single module, jqmc.atomic_orbital.
# The grad / Laplacian cross-check helpers (_compute_AOs_grad_autodiff,
# _compute_AOs_grad_debug, _compute_AOs_laplacian_autodiff,
# _compute_AOs_laplacian_debug) and the forward oracle (_compute_AOs_debug) are
# leaf-agnostic: they route through the public ``compute_AOs`` wrapper, so the
# same functions cover the GTO, plane-wave and B-spline leaves unchanged.
from jqmc._precision import (
    get_complex_dtype_np,
    get_dtype_jnp,
    get_dtype_np,
    get_evaluation_kappa_max,
    get_tolerance,
    get_tolerance_min,
    is_mixed_precision_enabled,
)
from jqmc._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_AO_IMAGE_TRUNCATION_SUM_FACTOR,
    ATOL_AO_OVERLAP_GRID_QUADRATURE,
    ATOL_BIT_EXACT,
    ATOL_FLOOR_MEDIUM,
    ATOL_INTERPOLATION_RELAXED,
    ATOL_INTERPOLATION_TIGHT,
    ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_MEDIUM,
    get_tolerance_mixed_fidelity,
    RTOL_BIT_EXACT,
    RTOL_PBC_SEAM_CONTINUITY,
    RTOL_PGTO_LATTICE_OVERLAP_CONVERGENCE_FACTOR,
    RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    get_pgto_lattice_overlap_term_tolerance,
)
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_bspline_data,
    AOs_bspline_tiled_complex_data,
    AOs_cart_gto_data,
    AOs_cart_pgto_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    _apply_image_expansion,
    _apply_image_expansion_complex,
    _build_ragged_image_layout,
    _compute_AOs_cart_debug,
    _compute_AOs_debug,
    _compute_AOs_grad_autodiff,
    _compute_AOs_grad_debug,
    _compute_AOs_laplacian_autodiff,
    _compute_AOs_laplacian_debug,
    _compute_AOs_sphe_debug,
    _compute_overlap_matrix_debug,
    _compute_overlap_matrix_periodic_pgto_lattice,
    _compute_S_l_m,
    _compute_S_l_m_debug,
    _detile_ragged_to_molecular,
    _fold_image_axis_ragged,
    _promote_pgto_to_complex,
    _with_periodic_image_expansion_debug,
    build_AOs_cart_pgto_complex_data,
    build_AOs_sphe_pgto_complex_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
    compute_overlap_matrix,
)
from jqmc.molecular_orbital import MOs_data, compute_MOs

# Spline prefiltering lives in the projection module (not atomic_orbital); only
# the B-spline leaf-specific "reproduces smooth bands" tests use it.
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data, tile_structure_data
from jqmc.trexio_wrapper import read_trexio_file
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from tests.conftest import CELL_FIXTURES, make_pbc_structure

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _assert_orbital_derivative_mixed_fidelity(
    mixed: np.ndarray,
    full: np.ndarray,
    atol: float,
    rtol: float,
) -> None:
    """Require one orbital-derivative block to satisfy its mixed error model."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


# Seeded evaluation-point screen of the spherical/solid-harmonics oracle test
# One generator stream per (l, m); every accepted point is compared.
_SH_EVALUATION_SEED = 20261002
_SH_ACCEPTED_POINTS = 32
_SH_MAXIMUM_DRAWS = 64
# Deterministic rejection counts per (l, m) for the central cap
# rtol / ((4 l + 4) eps) of the ao_eval strict comparison (absent key: zero).
# Full precision (fp64, cap >= 1e8) rejects nothing; mixed (fp32) rejects only
# draws whose summed expansion cancels by more than the cap allows.
_SH_EXPECTED_REJECTIONS = {
    False: {},
    True: {
        (3, -1): 1,
        (4, -1): 1,
        (4, 0): 2,
        (4, 1): 1,
        (4, 2): 1,
        (5, -3): 1,
        (5, -1): 6,
        (5, 0): 6,
        (5, 2): 3,
        (5, 3): 1,
        (5, 4): 2,
        (5, 5): 1,
        (6, -5): 1,
        (6, -3): 2,
        (6, -2): 5,
        (6, -1): 2,
        (6, 0): 11,
        (6, 2): 12,
        (6, 4): 2,
        (6, 5): 1,
        (6, 6): 1,
    },
}


def _solid_harmonic_evaluation_kappa(l, m, R_cart, r_cart):
    """Evaluation condition number of the summed solid-harmonic expansion.

    ``_compute_S_l_m_debug`` evaluates ``N * Lambda * A`` with
    ``Lambda = sum_k lambda_k r^(2k) z^(l-2k-|m|)`` and
    ``A = sum_p binom(|m|, p) phase_p x^p y^(|m|-p)``. For that product of sums
    ``kappa = (sum |Lambda terms|) (sum |A terms|) / |Lambda A|`` bounds the
    relative evaluation error by ``kappa * c_l * eps`` (see
    ``jqmc._setting.evaluation_kappa_max``). It is computed here in the same
    ``ao_eval`` zone dtype as the evaluation it screens; the normalization N
    cancels. A zero value gives an infinite kappa.
    """
    dtype = get_dtype_np("ao_eval")
    x, y, z = np.asarray(np.asarray(r_cart) - np.asarray(R_cart), dtype=dtype)
    radius = np.sqrt(x * x + y * y + z * z)
    m_abs = abs(m)
    polar = [
        dtype(
            (-1.0) ** k
            * 2.0 ** (-l)
            * math.comb(l, k)
            * math.comb(2 * l - 2 * k, l)
            * math.factorial(l - 2 * k)
            / math.factorial(l - 2 * k - m_abs)
        )
        * radius ** (2 * k)
        * z ** (l - 2 * k - m_abs)
        for k in range((l - m_abs) // 2 + 1)
    ]
    phase = (1, 0, -1, 0) if m >= 0 else (0, 1, 0, -1)
    azimuth = [dtype(math.comb(m_abs, p) * phase[(m_abs - p) % 4]) * x**p * y ** (m_abs - p) for p in range(m_abs + 1)]
    value = abs(sum(polar) * sum(azimuth))
    bound = sum(abs(term) for term in polar) * sum(abs(term) for term in azimuth)
    return np.inf if value == 0 else bound / value


class TestSphericalHarmonics:
    @pytest.mark.parametrize(
        ["l", "m"],
        list(
            itertools.chain.from_iterable(
                [[pytest.param(l, m, id=f"l={l}, m={m}") for m in range(-l, l + 1)] for l in range(7)]
            )
        ),
    )
    @pytest.mark.light
    def test_spherical_harmonics_debug_vs_production(self, l, m):
        """Test the spherical harmonics."""

        def Y_l_m_ref(l=0, m=0, r_cart_rel=None):
            if r_cart_rel is None:
                r_cart_rel = [0.0, 0.0, 0.0]
            """See https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics"""
            x, y, z = r_cart_rel[..., 0], r_cart_rel[..., 1], r_cart_rel[..., 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            # s orbital
            if (l, m) == (0, 0):
                return 1.0 / 2.0 * np.sqrt(1.0 / np.pi) * r**0.0
            # p orbitals
            if (l, m) == (1, -1):
                return np.sqrt(3.0 / (4 * np.pi)) * y / r
            if (l, m) == (1, 0):
                return np.sqrt(3.0 / (4 * np.pi)) * z / r
            if (l, m) == (1, 1):
                return np.sqrt(3.0 / (4 * np.pi)) * x / r
            # d orbitals
            if (l, m) == (2, -2):
                return 1.0 / 2.0 * np.sqrt(15.0 / (np.pi)) * x * y / r**2
            if (l, m) == (2, -1):
                return 1.0 / 2.0 * np.sqrt(15.0 / (np.pi)) * y * z / r**2
            if (l, m) == (2, 0):
                return 1.0 / 4.0 * np.sqrt(5.0 / (np.pi)) * (3 * z**2 - r**2) / r**2
            if (l, m) == (2, 1):
                return 1.0 / 2.0 * np.sqrt(15.0 / (np.pi)) * x * z / r**2
            if (l, m) == (2, 2):
                return 1.0 / 4.0 * np.sqrt(15.0 / (np.pi)) * (x**2 - y**2) / r**2
            # f orbitals
            if (l, m) == (3, -3):
                return 1.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * y * (3 * x**2 - y**2) / r**3
            if (l, m) == (3, -2):
                return 1.0 / 2.0 * np.sqrt(105.0 / (np.pi)) * x * y * z / r**3
            if (l, m) == (3, -1):
                return 1.0 / 4.0 * np.sqrt(21.0 / (2 * np.pi)) * y * (5 * z**2 - r**2) / r**3
            if (l, m) == (3, 0):
                return 1.0 / 4.0 * np.sqrt(7.0 / (np.pi)) * (5 * z**3 - 3 * z * r**2) / r**3
            if (l, m) == (3, 1):
                return 1.0 / 4.0 * np.sqrt(21.0 / (2 * np.pi)) * x * (5 * z**2 - r**2) / r**3
            if (l, m) == (3, 2):
                return 1.0 / 4.0 * np.sqrt(105.0 / (np.pi)) * (x**2 - y**2) * z / r**3
            if (l, m) == (3, 3):
                return 1.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * x * (x**2 - 3 * y**2) / r**3
            # g orbitals
            if (l, m) == (4, -4):
                return 3.0 / 4.0 * np.sqrt(35.0 / (np.pi)) * x * y * (x**2 - y**2) / r**4
            if (l, m) == (4, -3):
                return 3.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * y * z * (3 * x**2 - y**2) / r**4
            if (l, m) == (4, -2):
                return 3.0 / 4.0 * np.sqrt(5.0 / (np.pi)) * x * y * (7 * z**2 - r**2) / r**4
            if (l, m) == (4, -1):
                return 3.0 / 4.0 * np.sqrt(5.0 / (2 * np.pi)) * y * (7 * z**3 - 3 * z * r**2) / r**4
            if (l, m) == (4, 0):
                return 3.0 / 16.0 * np.sqrt(1.0 / (np.pi)) * (35 * z**4 - 30 * z**2 * r**2 + 3 * r**4) / r**4
            if (l, m) == (4, 1):
                return 3.0 / 4.0 * np.sqrt(5.0 / (2 * np.pi)) * x * (7 * z**3 - 3 * z * r**2) / r**4
            if (l, m) == (4, 2):
                return 3.0 / 8.0 * np.sqrt(5.0 / (np.pi)) * (x**2 - y**2) * (7 * z**2 - r**2) / r**4
            if (l, m) == (4, 3):
                return 3.0 / 4.0 * np.sqrt(35.0 / (2 * np.pi)) * x * z * (x**2 - 3 * y**2) / r**4
            if (l, m) == (4, 4):
                return 3.0 / 16.0 * np.sqrt(35.0 / (np.pi)) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2)) / r**4
            if (l, m) == (5, -5):
                return 3.0 / 16.0 * np.sqrt(77.0 / (2 * np.pi)) * (5 * x**4 * y - 10 * x**2 * y**3 + y**5) / r**5
            if (l, m) == (5, -4):
                return 3.0 / 16.0 * np.sqrt(385.0 / np.pi) * 4 * x * y * z * (x**2 - y**2) / r**5
            if (l, m) == (5, -3):
                return 1.0 / 16.0 * np.sqrt(385.0 / (2 * np.pi)) * -1 * (y**3 - 3 * x**2 * y) * (9 * z**2 - r**2) / r**5
            if (l, m) == (5, -2):
                return 1.0 / 8.0 * np.sqrt(1155 / np.pi) * 2 * x * y * (3 * z**3 - z * r**2) / r**5
            if (l, m) == (5, -1):
                return 1.0 / 16.0 * np.sqrt(165 / np.pi) * y * (21 * z**4 - 14 * z**2 * r**2 + r**4) / r**5
            if (l, m) == (5, 0):
                return 1.0 / 16.0 * np.sqrt(11 / np.pi) * (63 * z**5 - 70 * z**3 * r**2 + 15 * z * r**4) / r**5
            if (l, m) == (5, 1):
                return 1.0 / 16.0 * np.sqrt(165 / np.pi) * x * (21 * z**4 - 14 * z**2 * r**2 + r**4) / r**5
            if (l, m) == (5, 2):
                return 1.0 / 8.0 * np.sqrt(1155 / np.pi) * (x**2 - y**2) * (3 * z**3 - z * r**2) / r**5
            if (l, m) == (5, 3):
                return 1.0 / 16.0 * np.sqrt(385.0 / (2 * np.pi)) * (x**3 - 3 * x * y**2) * (9 * z**2 - r**2) / r**5
            if (l, m) == (5, 4):
                return (
                    3.0 / 16.0 * np.sqrt(385.0 / np.pi) * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2)) / r**5
                )
            if (l, m) == (5, 5):
                return 3.0 / 16.0 * np.sqrt(77.0 / (2 * np.pi)) * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4) / r**5
            if (l, m) == (6, -6):
                return 1.0 / 64.0 * np.sqrt(6006.0 / np.pi) * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5) / r**6
            if (l, m) == (6, -5):
                return 3.0 / 32.0 * np.sqrt(2002.0 / np.pi) * z * (5 * x**4 * y - 10 * x**2 * y**3 + y**5) / r**6
            if (l, m) == (6, -4):
                return 3.0 / 32.0 * np.sqrt(91.0 / np.pi) * 4 * x * y * (11 * z**2 - r**2) * (x**2 - y**2) / r**6
            if (l, m) == (6, -3):
                return 1.0 / 32.0 * np.sqrt(2730.0 / np.pi) * -1 * (11 * z**3 - 3 * z * r**2) * (y**3 - 3 * x**2 * y) / r**6
            if (l, m) == (6, -2):
                return 1.0 / 64.0 * np.sqrt(2730.0 / np.pi) * 2 * x * y * (33 * z**4 - 18 * z**2 * r**2 + r**4) / r**6
            if (l, m) == (6, -1):
                return 1.0 / 16.0 * np.sqrt(273.0 / np.pi) * y * (33 * z**5 - 30 * z**3 * r**2 + 5 * z * r**4) / r**6
            if (l, m) == (6, 0):
                return (
                    1.0 / 32.0 * np.sqrt(13.0 / np.pi) * (231 * z**6 - 315 * z**4 * r**2 + 105 * z**2 * r**4 - 5 * r**6) / r**6
                )
            if (l, m) == (6, 1):
                return 1.0 / 16.0 * np.sqrt(273.0 / np.pi) * x * (33 * z**5 - 30 * z**3 * r**2 + 5 * z * r**4) / r**6
            if (l, m) == (6, 2):
                return 1.0 / 64.0 * np.sqrt(2730.0 / np.pi) * (x**2 - y**2) * (33 * z**4 - 18 * z**2 * r**2 + r**4) / r**6
            if (l, m) == (6, 3):
                return 1.0 / 32.0 * np.sqrt(2730.0 / np.pi) * (11 * z**3 - 3 * z * r**2) * (x**3 - 3 * x * y**2) / r**6
            if (l, m) == (6, 4):
                return (
                    3.0
                    / 32.0
                    * np.sqrt(91.0 / np.pi)
                    * (11 * z**2 - r**2)
                    * (x**2 * (x**2 - 3 * y**2) + y**2 * (y**2 - 3 * x**2))
                    / r**6
                )
            if (l, m) == (6, 5):
                return 3.0 / 32.0 * np.sqrt(2002.0 / np.pi) * z * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4) / r**6
            if (l, m) == (6, 6):
                return 1.0 / 64.0 * np.sqrt(6006.0 / np.pi) * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6) / r**6
            raise NotImplementedError

        # Seeded evaluation points with a deterministic evaluation-condition
        # screen. Near an angular node the
        # summed expansion cancels, and no fixed-precision evaluation can meet
        # the unchanged rtol there; draws whose kappa exceeds the central cap
        # rtol / ((4 l + 4) eps) are rejected and redrawn from the same stream.
        R_cart = [0.0, 0.0, 1.0]
        r_cart_min, r_cart_max = -10.0, 10.0
        rng = np.random.default_rng([_SH_EVALUATION_SEED, l, m + l])
        kappa_max = get_evaluation_kappa_max("ao_eval", "strict", l)
        atol, rtol = get_tolerance("ao_eval", "strict")
        accepted_draws = 0
        rejected_draws = 0
        for _draw in range(_SH_MAXIMUM_DRAWS):
            r_cart = rng.uniform(r_cart_min, r_cart_max, size=3)
            evaluation_kappa = _solid_harmonic_evaluation_kappa(l, m, R_cart, r_cart)
            if not evaluation_kappa <= kappa_max:
                rejected_draws += 1
                continue
            assert evaluation_kappa <= kappa_max
            accepted_draws += 1
            r_norm = LA.norm(np.array(R_cart) - np.array(r_cart))
            r_cart_rel = np.array(r_cart) - np.array(R_cart)
            test_S_lm = _compute_S_l_m_debug(
                atomic_center_cart=R_cart,
                angular_momentum=l,
                magnetic_quantum_number=m,
                r_cart=r_cart,
            )
            ref_S_lm = np.sqrt((4 * np.pi) / (2 * l + 1)) * r_norm**l * Y_l_m_ref(l=l, m=m, r_cart_rel=r_cart_rel)
            assert not np.any(np.isnan(np.asarray(test_S_lm))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(ref_S_lm))), "NaN detected in second argument"
            assert_allclose(test_S_lm, ref_S_lm, atol=atol, rtol=rtol)
            if accepted_draws == _SH_ACCEPTED_POINTS:
                break
        # Enough accepted points remain for every (l, m), and the realized
        # rejection count is the recorded deterministic one for this mode.
        assert accepted_draws == _SH_ACCEPTED_POINTS
        assert rejected_draws == _SH_EXPECTED_REJECTIONS[is_mixed_precision_enabled()].get((l, m), 0)

        jax.clear_caches()

    @pytest.mark.light
    def test_solid_harmonics_debug_vs_production(self):
        """Test the solid harmonics with a batch."""
        seed = 34487
        np.random.seed(seed)

        num_R_cart_samples = 49  # fixed
        num_r_cart_samples = 10
        r_cart_min, r_cart_max = -1.0, 1.0
        R_cart_min, R_cart_max = 0.0, 0.0
        r_carts = (r_cart_max - r_cart_min) * np.random.rand(num_r_cart_samples, 3) + r_cart_min
        R_carts = (R_cart_max - R_cart_min) * np.random.rand(num_R_cart_samples, 3) + R_cart_min
        r_R_diffs_uq = r_carts[None, :, :] - R_carts[:, None, :]

        ml_list = list(itertools.chain.from_iterable([[(l, m) for m in range(-l, l + 1)] for l in range(7)]))

        # S_l_m debug
        S_l_m_debug = np.array(
            [
                [
                    [
                        _compute_S_l_m_debug(
                            angular_momentum=l, magnetic_quantum_number=m, atomic_center_cart=R_cart, r_cart=r_cart
                        )
                        for r_cart in r_carts
                    ]
                    for R_cart in R_carts
                ]
                for l, m in ml_list
            ]
        )

        # S_l_m jax
        _, S_l_m_jax = _compute_S_l_m(r_R_diffs_uq)

        atol, rtol = get_tolerance("ao_eval", "strict")
        assert not np.any(np.isnan(np.asarray(S_l_m_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(S_l_m_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(S_l_m_debug, S_l_m_jax, atol=atol, rtol=rtol)
        jax.clear_caches()


def _make_aos_data(
    structure_data: Structure_data,
    ao_kind: str,
    *,
    l_max: int = 1,
    uniform: bool = True,
) -> AOs_cart_gto_data | AOs_sphe_gto_data:
    """Build a contracted AO container covering full angular-momentum shells.

    Generates one AO per ``(l, m)`` for sphe and one AO per
    ``(l, nx, ny, nz)`` with ``nx + ny + nz == l`` for cart, for every
    ``l in {0, ..., l_max}``. Each AO is contracted from a fixed number
    of primitives ``K`` (see ``uniform`` below) drawn from base exponents
    ``[1.0, 0.5, 2.0]`` and coefficients ``[1.0, 0.6, 0.3]``.

    ``nucleus_index`` is auto-derived from ``structure_data``:

      * 1 atom => all AOs centered on atom 0 (typical PBC fixture);
      * ``num_ao`` atoms => one atom per AO (the rich-basis molecular
        layout of the kernel-level tests).

    Both call sites use this:

      * PBC tests pass ``l_max=3, uniform=True`` on a single-atom
        structure -- ``K = 1`` everywhere yields full s..f shells on one
        atom. The image set is dominated by ``(alpha, L) = (1.0, 1)``
        (L >= 2 have smaller ``r_cut`` and their image sets are subsets);
        the union determines ``_n_image`` on the cubic 10 Bohr fixture,
        whose exact value depends on ``AO_IMAGE_TRUNCATION_ATOL`` (see
        :py:func:`test_compute_AOs_public_vs_debug_periodic`, which derives
        the expectation from the helper-free oracle). ``l_max`` is capped
        at 3 (not 6) so the quadratic ``shell_indices`` construction in the
        kernel-level debug oracles stays tractable on the expanded
        ``num_ao = num_ao_molecular * _n_image`` array.
      * Molecular tests pass ``l_max=6, uniform=False`` -- chemistry-like
        contraction with ``K = 3`` for s, ``K = 2`` for p, and ``K = 1``
        for ``l >= 2``. The 3 distinct ``K`` values force the
        multi-bucket general path of
        :py:func:`jqmc.atomic_orbital._reduce_primitives_to_aos`
        (uniform ``K`` would fall into the single-bucket fast path and
        skip that code).

    Args:
        structure_data: Parent structure; ``pbc_flag`` is honoured
            downstream and the atom count selects the ``nucleus_index``
            layout (see above).
        ao_kind: ``"cart"`` or ``"sphe"``.
        l_max: Maximum angular momentum included. AOs cover all
            ``l in 0..l_max``.
        uniform: Contraction depth policy.

          * ``True`` (default) -- every AO has ``K = 1`` (single primitive
            per AO). Exercises the single-bucket fast path of
            ``_reduce_primitives_to_aos``.
          * ``False`` -- chemistry-like multi-K: ``K = 3`` for ``l = 0``,
            ``K = 2`` for ``l = 1``, ``K = 1`` for ``l >= 2``. Exercises
            the multi-bucket general path.
    """
    # The first base_exponents / base_coefficients slot pins alpha = 1.0 /
    # coeff = 1.0 so the PBC ``_n_image`` derivation in
    # test_compute_AOs_public_vs_debug_periodic (oracle-driven) stays exact
    # under the unified helper.
    base_exponents = [1.0, 0.5, 2.0]
    base_coefficients = [1.0, 0.6, 0.3]

    # AO catalog over l = 0..l_max.
    if ao_kind == "sphe":
        ml_list = list(itertools.chain.from_iterable([(l, m) for m in range(-l, l + 1)] for l in range(l_max + 1)))
        angular_momentums = tuple(l for l, _ in ml_list)
        magnetic_quantum_numbers = tuple(m for _, m in ml_list)
        num_ao = len(ml_list)
        cart_extras = None
    elif ao_kind == "cart":
        am_list: list[int] = []
        nx_list: list[int] = []
        ny_list: list[int] = []
        nz_list: list[int] = []
        for l in range(l_max + 1):
            for po in itertools.combinations_with_replacement("xyz", l):
                po_s = "".join(po)
                am_list.append(l)
                nx_list.append(po_s.count("x"))
                ny_list.append(po_s.count("y"))
                nz_list.append(po_s.count("z"))
        angular_momentums = tuple(am_list)
        cart_extras = (tuple(nx_list), tuple(ny_list), tuple(nz_list))
        num_ao = len(am_list)
        magnetic_quantum_numbers = None
    else:
        raise ValueError(f"unknown ao_kind: {ao_kind!r} (expected 'cart' or 'sphe')")

    # Per-AO contraction depth.
    if uniform:
        k_per_ao = [1] * num_ao
    else:
        # Multi-K policy: K = 3 for s, K = 2 for p, K = 1 for higher l.
        # 3 distinct K values are required to force the multi-bucket path
        # of _reduce_primitives_to_aos.
        k_per_ao = [(3 if l == 0 else 2 if l == 1 else 1) for l in angular_momentums]

    # nucleus_index layout follows structure_data atom count.
    n_atoms = len(structure_data.atomic_numbers)
    if n_atoms == 1:
        nucleus_index = tuple([0] * num_ao)
    elif n_atoms == num_ao:
        nucleus_index = tuple(range(num_ao))
    else:
        raise ValueError(
            f"structure_data has {n_atoms} atoms; expected 1 (all AOs on one "
            f"atom) or num_ao={num_ao} (one atom per AO) for "
            f"l_max={l_max}, ao_kind={ao_kind!r}."
        )

    # Contraction: per-AO K primitives, group-by-AO ``orbital_indices`` layout.
    # Primitives of AO ``i`` occupy positions
    # ``sum(k_per_ao[:i]) .. sum(k_per_ao[:i]) + k_per_ao[i]``.
    orbital_indices_list: list[int] = []
    exponents_list: list[float] = []
    coefficients_list: list[float] = []
    for ao_idx in range(num_ao):
        K_ao = k_per_ao[ao_idx]
        orbital_indices_list.extend([ao_idx] * K_ao)
        exponents_list.extend(base_exponents[:K_ao])
        coefficients_list.extend(base_coefficients[:K_ao])
    orbital_indices = tuple(orbital_indices_list)
    exponents = np.array(exponents_list, dtype=np.float64)
    coefficients = np.array(coefficients_list, dtype=np.float64)
    num_ao_prim = sum(k_per_ao)

    common = dict(
        structure_data=structure_data,
        nucleus_index=nucleus_index,
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
    )
    if ao_kind == "sphe":
        return AOs_sphe_gto_data._from_full_basis(
            **common,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
        )
    return AOs_cart_gto_data._from_full_basis(
        **common,
        polynominal_order_x=cart_extras[0],
        polynominal_order_y=cart_extras[1],
        polynominal_order_z=cart_extras[2],
    )


class TestComputeAOs:
    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_compute_AOs_public_vs_debug_molecular(self, ao_kind: str):
        """Public ``compute_AOs`` matches ``_compute_AOs_debug`` on a rich
        multi-K-contracted molecular basis (l = 0..6).

        Public-layer comparison over the full s..h angular-momentum range
        on one atom per AO at random positions in [-1, 1] Bohr with a
        ~100-electron batch, and additionally exercises the
        segment-sum-over-primitives contraction reduction with **per-shell
        multi-K** contraction depths (s: K=3, p: K=2, l >= 2: K=1).

        The 3 distinct K values force
        :py:func:`jqmc.atomic_orbital._reduce_primitives_to_aos` to take the
        multi-bucket general path (concat-of-per-K-reduce + inverse-permutation
        gather). Uniform-K bases would fall into the single-bucket fast path
        and skip this code.

        Confirms:
          * the public-API ``isinstance`` dispatch shim routes cart and sphe
            to the matching kernel-level functions,
          * the ``_n_image == 1`` short-circuit in both ``compute_AOs`` and
            ``_compute_AOs_debug`` forwards the kernel-level result unchanged,
          * production and debug agree on the multi-K contraction reduction
            across every ``(l, m)`` / ``(l, nx, ny, nz)`` for ``l <= 6``.
        """
        rng = np.random.default_rng(20260602)

        # Atom count = num_ao (one atom per AO) -- the rich-basis layout of
        # the kernel-level tests.
        if ao_kind == "sphe":
            num_ao = sum(2 * l + 1 for l in range(7))  # 49
        else:
            num_ao = sum((l + 1) * (l + 2) // 2 for l in range(7))  # 84

        positions = rng.uniform(-1.0, 1.0, size=(num_ao, 3)).astype(np.float64)
        structure_mol = Structure_data(
            pbc_flag=False,
            positions=positions,
            atomic_numbers=tuple([0] * num_ao),
            element_symbols=tuple(["X"] * num_ao),
            atomic_labels=tuple(["X"] * num_ao),
        )
        structure_mol.sanity_check()

        # uniform=False -> the helper's chemistry-like multi-K policy
        # (K = 3 for s, K = 2 for p, K = 1 for l >= 2; 3 distinct K values
        # force the multi-bucket general path of _reduce_primitives_to_aos).
        aos_mol = _make_aos_data(structure_mol, ao_kind, l_max=6, uniform=False)
        aos_mol.sanity_check()
        assert aos_mol._n_image == 1, f"molecular {type(aos_mol).__name__} must have _n_image == 1"
        assert aos_mol.num_ao == num_ao
        # Multi-K invariant: confirm the basis actually has >= 2 distinct K
        # values so the multi-bucket reduction path is genuinely exercised.
        # k_per_ao reconstructed by counting primitives per AO.
        _orb = np.asarray(aos_mol.orbital_indices, dtype=np.int64)
        _k_observed = sorted(set(np.bincount(_orb, minlength=num_ao).tolist()))
        assert len(_k_observed) >= 2, (
            f"_make_aos_data should yield multi-K basis with the chemistry policy; observed K values: {_k_observed}"
        )

        num_el = 100
        r_carts = rng.uniform(-1.0, 1.0, size=(num_el, 3)).astype(np.float64)

        prod = np.asarray(compute_AOs(aos_data=aos_mol, r_carts=r_carts))
        ref = np.asarray(_compute_AOs_debug(aos_data=aos_mol, r_carts=r_carts))

        # Shape contract: molecular path => (num_ao, N_e).
        assert prod.shape == (num_ao, num_el)
        assert ref.shape == prod.shape

        assert not np.any(np.isnan(prod)), f"NaN in production compute_AOs ({ao_kind})"
        assert not np.any(np.isnan(ref)), f"NaN in _compute_AOs_debug oracle ({ao_kind})"

        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(prod, ref, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize(
        "entry",
        ["sanity_check", "compute_AOs", "compute_AOs_grad", "compute_AOs_laplacian", "compute_AOs_value_grad_lap"],
    )
    def test_sphe_l_above_6_is_rejected(self, entry: str):
        """A spherical basis with an l = 7 shell fails loudly on every entry point.

        The XLA real-solid-harmonic tables stop at l = 6 (49 (l, m) rows).
        Before the guard, an l = 7 basis passed ``sanity_check`` and the
        kernels silently returned O(1)-wrong l = 7 AOs; with a second nucleus
        the per-nucleus stride (64 assumed vs 49 tabulated) also corrupted the
        s/p/d AOs of that nucleus. Two nuclei are used so both failure modes
        are reachable. The kernel entries are called without ``sanity_check``
        so the kernel-side guard is checked on its own.
        """
        structure = Structure_data(
            pbc_flag=False,
            positions=np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.1]]),
            atomic_numbers=(6, 6),
            element_symbols=("X", "X"),
            atomic_labels=("X", "X"),
        )
        shells = [(0, l) for l in range(8)] + [(1, l) for l in range(3)]
        nucleus_index, exponents, angular_momentums, magnetic = [], [], [], []
        for a, l in shells:
            for m in range(-l, l + 1):
                nucleus_index.append(a)
                exponents.append(0.6 + 0.3 * l)
                angular_momentums.append(l)
                magnetic.append(m)
        num_ao = len(nucleus_index)
        aos = AOs_sphe_gto_data._from_full_basis(
            exponents=np.array(exponents),
            coefficients=np.ones(num_ao),
            structure_data=structure,
            nucleus_index=tuple(nucleus_index),
            num_ao=num_ao,
            num_ao_prim=num_ao,
            orbital_indices=tuple(range(num_ao)),
            angular_momentums=tuple(angular_momentums),
            magnetic_quantum_numbers=tuple(magnetic),
        )
        r_carts = np.random.default_rng(3).uniform(-1.5, 1.5, size=(6, 3))
        calls = {
            "sanity_check": lambda: aos.sanity_check(),
            "compute_AOs": lambda: compute_AOs(aos, r_carts),
            "compute_AOs_grad": lambda: compute_AOs_grad(aos, r_carts),
            "compute_AOs_laplacian": lambda: compute_AOs_laplacian(aos, r_carts),
            "compute_AOs_value_grad_lap": lambda: compute_AOs_value_grad_lap(aos, r_carts),
        }
        with pytest.raises(ValueError, match=r"l <= 6"):
            jax.block_until_ready(calls[entry]())
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_compute_AOs_public_vs_debug_periodic(self, ao_kind: str):
        """Public ``compute_AOs`` matches ``_compute_AOs_debug`` on a PBC instance.

        Exercises the post-kernel image-axis reduction (``reshape(n_image, M_c,
        -1).sum(0)``) on *both* sides of the comparison: the production wrapper
        inside ``compute_AOs`` and the matching reduction added to
        ``_compute_AOs_debug``. Confirms the two paths fold the
        image axis in the same order and produce identical periodised AO values
        at the public-API rank ``(num_ao_molecular, N_e)``. Parametrised over
        both AO kinds; the spherical leaf additionally exercises the solid-
        harmonics centre accessor that is unique to the spherical kernel.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        structure_pbc.sanity_check()
        aos_pbc_mol = _make_aos_data(structure_pbc, ao_kind, l_max=3, uniform=True)
        aos_pbc_mol.sanity_check()

        expanded_aos = _apply_image_expansion(
            aos_pbc_mol,
            structure_pbc,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        # The expected image count depends on AO_IMAGE_TRUNCATION_ATOL (looser
        # tol => smaller r_cut => fewer images), so derive it from the helper-free
        # oracle ``_with_periodic_image_expansion_debug`` rather than hardcoding a
        # value tied to one particular tolerance. Cross-checking the production
        # helper chain against the oracle is the intended ``_debug`` discipline.
        expected_aos = _with_periodic_image_expansion_debug(
            aos_pbc_mol,
            structure_pbc,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        assert expanded_aos._n_image == expected_aos._n_image, (
            f"production _n_image ({expanded_aos._n_image}) must match the "
            f"oracle ({expected_aos._n_image}) on cubic 10 Bohr / s..f shells "
            f"({ao_kind}) / alpha=1.0 / "
            f"AO_IMAGE_TRUNCATION_ATOL={AO_IMAGE_TRUNCATION_ATOL}"
        )
        assert expanded_aos.num_ao == aos_pbc_mol.num_ao

        rng = np.random.default_rng(20260602)
        r_carts = rng.uniform(0.5, 9.5, size=(5, 3)).astype(np.float64)

        prod = np.asarray(compute_AOs(aos_data=expanded_aos, r_carts=r_carts))
        ref = np.asarray(_compute_AOs_debug(aos_data=expanded_aos, r_carts=r_carts))

        # Shape contract: PBC path collapses image axis => (num_ao, N_e).
        assert prod.shape == (expanded_aos.num_ao, r_carts.shape[0])
        assert ref.shape == prod.shape

        assert not np.any(np.isnan(prod)), f"NaN in production compute_AOs (PBC, {ao_kind})"
        assert not np.any(np.isnan(ref)), f"NaN in _compute_AOs_debug oracle (PBC, {ao_kind})"

        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(prod, ref, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_pgto_image_indices_roundtrip(self, ao_kind: str):
        """The pgto leaf stores integer image indices and the
        derived ``_expanded_positions_cart_*`` properties reconstruct the same
        Cartesian centres ``R_mol[I] + (m . cell)[n]`` regardless of whether
        the lattice is accessed in numpy or JAX.

        Pins three contracts:

          1. ``_image_indices`` is a hashable tuple-of-triples whose count
             equals ``_n_image``.
          2. ``np`` and ``jnp`` properties agree to fp64 round-off (they share
             the same integer index source; only the cell access dtype
             differs).
          3. The reconstructed Cartesian centres are bit-equal to the direct
             ``n . cell`` formula (the properties evaluate it lazily on each
             access).
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos_pbc_mol = _make_aos_data(structure_pbc, ao_kind, l_max=1, uniform=True)
        expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_pbc, tol=AO_IMAGE_TRUNCATION_ATOL)

        # 1. _image_indices holds ONE int triple per kernel-AO (ragged layout).
        assert isinstance(expanded_aos._image_indices, tuple)
        assert len(expanded_aos._image_indices) == expanded_aos._num_ao_kernel
        for row in expanded_aos._image_indices:
            assert isinstance(row, tuple) and len(row) == 3
            assert all(isinstance(c, int) for c in row)

        # 2. np vs jnp agree (same integer source, same cell, just dtype).
        exp_np = np.asarray(expanded_aos._expanded_positions_cart_np)
        exp_jnp = np.asarray(expanded_aos._expanded_positions_cart_jnp)
        np.testing.assert_allclose(exp_np, exp_jnp, atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_BIT_EXACT)

        # 3. Bit-equal to the per-kernel-AO centre R_{atom(k)} + T_{image(k)}.
        n = np.asarray(expanded_aos._image_indices, dtype=np.float64).reshape(-1, 3)
        T = n @ structure_pbc.cell  # (M, 3)
        R = np.asarray(structure_pbc.positions, dtype=np.float64)  # (natom, 3)
        atom = np.asarray(expanded_aos._ragged_kernel_atom, dtype=np.int64)  # (M,)
        exp_expected = R[atom] + T  # (M, 3)
        np.testing.assert_array_equal(exp_np, exp_expected)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_compute_AOs_periodic_reshape_sum_order(self, ao_kind: str):
        """The image-axis reshape-sum in ``compute_AOs`` collapses dim 0 of
        ``(n_image, num_ao_molecular, N_e)`` -- NOT dim 1.

        Independent of the parallel reshape-sum in ``_compute_AOs_debug``: takes
        the raw ``(n_image * num_ao_molecular, N_e)`` kernel-level debug output
        (cart or sphe, selected by ``ao_kind`` to match the dataclass kind),
        manually folds it with the documented
        ``reshape(n_image, M_c, -1).sum(0)`` pattern, and compares to the
        public wrapper. Catches a hypothetical regression where the production
        wrapper's reduction axis is swapped or the layout invariant
        ``atom-major within image`` is broken on either dispatch leg.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos_pbc_mol = _make_aos_data(structure_pbc, ao_kind, l_max=3, uniform=True)
        expanded_aos = _apply_image_expansion(
            aos_pbc_mol,
            structure_pbc,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        assert expanded_aos._n_image > 1, "fixture must exercise the PBC fold path"

        rng = np.random.default_rng(20260602)
        r_carts = rng.uniform(0.5, 9.5, size=(5, 3)).astype(np.float64)

        kernel_debug = _compute_AOs_cart_debug if ao_kind == "cart" else _compute_AOs_sphe_debug
        raw = np.asarray(kernel_debug(aos_data=expanded_aos, r_carts=r_carts))
        # raw shape: (n_image * num_ao, N_e)
        assert raw.shape == (
            expanded_aos._n_image * expanded_aos.num_ao,
            r_carts.shape[0],
        )

        manual_fold = raw.reshape(
            expanded_aos._n_image,
            expanded_aos.num_ao,
            -1,
        ).sum(axis=0)
        prod = np.asarray(compute_AOs(aos_data=expanded_aos, r_carts=r_carts))

        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(prod, manual_fold, atol=atol, rtol=rtol)

        jax.clear_caches()

    # ===========================================================================
    # Non-GTO leaves: forward oracle + leaf-specific contracts.
    #
    # The grad / Laplacian three-way agreement (analytic vs autodiff vs numerical)
    # for the plane-wave and B-spline leaves is covered by the periodic
    # ``test_AOs_grad_*`` / ``test_AOs_laplacian_*`` tests above, which are
    # parametrised over ``ao_kind in {"cart", "sphe", "pw", "bspline"}`` and reuse
    # the leaf-agnostic ``_compute_AOs_grad_autodiff`` / ``_compute_AOs_grad_debug``
    # / ``_compute_AOs_laplacian_autodiff`` / ``_compute_AOs_laplacian_debug``
    # helpers from jqmc.atomic_orbital. The tests below add the forward oracle
    # cross-check (symmetric to ``test_compute_AOs_public_vs_debug_periodic``) and
    # the per-leaf structural contracts (sanity_check, complex guard, periodicity,
    # the agnostic MO path).
    # ===========================================================================

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["pw", "bspline"])
    def test_compute_AOs_public_vs_debug_nongto(self, ao_kind: str):
        """Public ``compute_AOs`` matches ``_compute_AOs_debug`` for the
        non-GTO leaves (plane-wave / B-spline).

        Symmetric to :py:func:`test_compute_AOs_public_vs_debug_periodic`: the
        production JAX kernel is cross-checked against the plain-numpy forward
        oracle that ``_compute_AOs_debug`` dispatches to
        (``_compute_AOs_plane_wave_debug`` / ``_compute_AOs_bspline_debug``).
        Non-GTO leaves carry no periodic-image axis, so both sides return
        ``(num_ao, N_e)`` directly.
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)
        prod = np.asarray(compute_AOs(aos_data=aos_data, r_carts=r_carts))
        ref = np.asarray(_compute_AOs_debug(aos_data=aos_data, r_carts=r_carts))
        assert prod.shape == (aos_data.num_ao, r_carts.shape[0])
        assert ref.shape == prod.shape
        assert not np.any(np.isnan(prod)), f"NaN in production compute_AOs ({ao_kind})"
        assert not np.any(np.isnan(ref)), f"NaN in _compute_AOs_debug oracle ({ao_kind})"
        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(prod, ref, atol=atol, rtol=rtol)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Non-GTO leaf scaffolding (plane-wave + B-spline).
# Self-contained small cubic PBC cell shared by the grad / Laplacian
# cross-checks (via _setup_grad_lap_test) and the leaf-specific tests below.
# All AO production / oracle helpers still come from jqmc.atomic_orbital; only
# the B-spline prefilter (bspline_prefilter_periodic)
# comes from jqmc.pw_pgto_projection.
# ---------------------------------------------------------------------------

_NONGTO_A = 6.0  # cubic cell edge (Bohr) for the PW / B-spline leaves
_BSPLINE_NS = (12, 12, 12)
_BSPLINE_ORDER = 5


def _nongto_cubic_structure(a: float = _NONGTO_A) -> Structure_data:
    """Single-atom cubic periodic cell for the PW / B-spline leaves."""
    cell = np.eye(3, dtype=np.float64) * a
    return Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3), dtype=np.float64),
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )


def _pw_basis(structure: Structure_data):
    """Real cos/sin plane-wave basis: constant + cos/sin for a few G > 0.

    Returns ``(g_vectors, phase_is_sin)`` with 1 constant + 5 (cos, sin)
    pairs = 11 basis functions.
    """
    recip = structure.recip_cell  # (3, 3) rows are reciprocal vectors (Bohr^-1)
    millers = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 1]],
        dtype=np.float64,
    )
    g_list: list[np.ndarray] = []
    is_sin: list[float] = []
    for m in millers:
        g = m @ recip
        if np.all(m == 0):
            g_list.append(g)
            is_sin.append(0.0)  # constant (cos(0) = 1)
        else:
            g_list.append(g)
            is_sin.append(0.0)  # cos(G.r)
            g_list.append(g)
            is_sin.append(1.0)  # sin(G.r)
    return np.asarray(g_list, dtype=np.float64), np.asarray(is_sin, dtype=np.float64)


def _build_pw_leaf(structure: Structure_data) -> AOs_plane_wave_data:
    """Construct the real cos/sin plane-wave leaf for ``structure``."""
    g_vectors, phase_is_sin = _pw_basis(structure)
    return AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)


def _pw_reference_values(g_vectors, phase_is_sin, r_carts):
    """Direct numpy cos/sin reference in the ``ao_eval`` zone, shape ``(num_ao, N_e)``.

    Precision: production evaluates the plane-wave values in the ``ao_eval`` zone
    (float32 in mixed). The geometric phase ``G.r`` is formed in float64 (the
    caller's geometry precision), cast to the ``ao_eval`` dtype, and cos/sin
    are evaluated in that dtype, so a mixed run is judged against a reference
    in its own precision regime. Every zone is float64 in full mode, where this
    is the float64 reference.
    """
    dtype = get_dtype_np("ao_eval")
    theta = (np.asarray(r_carts, dtype=np.float64) @ np.asarray(g_vectors, dtype=np.float64).T).astype(dtype)
    val = np.where(np.asarray(phase_is_sin)[None, :] > 0.5, np.sin(theta), np.cos(theta))
    return val.T


def _bspline_band_functions(frac):
    """Two smooth periodic bands of fractional coords, shape ``(n_band, ...)``.

    The arithmetic follows the dtype of ``frac``: the float64 spline-table
    construction in :py:func:`_build_bspline_leaf` stays float64, while the
    band-reproduction tests pass ``frac`` cast to the ``ao_eval`` dtype.
    """
    dtype = frac.dtype.type
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    tp = dtype(2.0 * np.pi)
    b0 = np.sin(tp * x) + dtype(0.5) * np.cos(tp * y) + dtype(0.25) * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + dtype(0.3) * np.sin(tp * z)
    return np.stack([b0, b1], axis=0)


def _bspline_band_reference_frac(structure: Structure_data, r_carts: np.ndarray) -> np.ndarray:
    """Home-cell fractional coordinates in float64, cast to the ``ao_eval`` dtype.

    Precision: the geometry (``r @ inv(cell)`` and the wrap into ``[0, 1)``) is
    reconstructed in float64; only the band arithmetic downstream runs in the
    ``ao_eval`` zone that evaluates the production spline values.
    """
    inv_cell = np.linalg.inv(np.asarray(structure.cell, dtype=np.float64))
    frac = (np.asarray(r_carts, dtype=np.float64) @ inv_cell) % 1.0
    return frac.astype(get_dtype_np("ao_eval"))


def _bspline_weights_zone(f: np.ndarray, order: int) -> np.ndarray:
    """Cardinal B-spline weights ``(n, order + 1)`` evaluated in the dtype of ``f``.

    Test-side twin of the float64 ``jqmc.pw_pgto_projection._bspline_weights``
    polynomials that keeps the input dtype, so the
    interpolation arithmetic can run in the ``ao_eval`` zone.
    """
    dtype = f.dtype.type
    one = dtype(1.0)
    if order == 3:
        w = [
            (one - f) ** 3,
            dtype(3.0) * f**3 - dtype(6.0) * f**2 + dtype(4.0),
            dtype(-3.0) * f**3 + dtype(3.0) * f**2 + dtype(3.0) * f + one,
            f**3,
        ]
        return np.stack(w, axis=-1) / dtype(6.0)
    if order == 5:
        w = [
            (one - f) ** 5,
            dtype(26.0) - dtype(50.0) * f + dtype(20.0) * f**2 + dtype(20.0) * f**3 - dtype(20.0) * f**4 + dtype(5.0) * f**5,
            dtype(66.0) - dtype(60.0) * f**2 + dtype(30.0) * f**4 - dtype(10.0) * f**5,
            dtype(26.0) + dtype(50.0) * f + dtype(20.0) * f**2 - dtype(20.0) * f**3 - dtype(20.0) * f**4 + dtype(10.0) * f**5,
            one + dtype(5.0) * f + dtype(10.0) * f**2 + dtype(10.0) * f**3 + dtype(5.0) * f**4 - dtype(5.0) * f**5,
            f**5,
        ]
        return np.stack(w, axis=-1) / dtype(120.0)
    raise ValueError(f"Unsupported B-spline order {order!r} (use 3 or 5).")


def _bspline_spline_reference_values(coeffs_cube: np.ndarray, frac: np.ndarray, order: int) -> np.ndarray:
    """Independent separable spline evaluation in the ``ao_eval`` complex dtype.

    Zone-precision replacement of the complex128 ``evaluate_spline_orbitals`` reference:
    the same spline table and the same separable local-support contraction
    (one Cartesian axis at a time, ``order + 1`` control points per axis), but
    the control points, spline coordinate, weights and contraction are all in
    ``get_complex_dtype_np("ao_eval")`` / ``get_dtype_np("ao_eval")``.
    ``frac`` is the float64-reconstructed home-cell fractional coordinate
    ``(n, 3)``. Returns the real part, shape ``(n_band, n)``.
    """
    real_dtype = get_dtype_np("ao_eval")
    coeffs = np.asarray(coeffs_cube).astype(get_complex_dtype_np("ao_eval"))
    spline_shape = coeffs.shape[1:]
    start = -((order - 1) // 2)
    offsets = np.arange(start, start + order + 1)
    frac_zone = np.asarray(frac, dtype=np.float64).astype(real_dtype)
    out = np.empty((coeffs.shape[0], frac_zone.shape[0]), dtype=real_dtype)
    for p in range(frac_zone.shape[0]):
        acc = coeffs
        for axis in range(3):
            t = frac_zone[p, axis] * real_dtype(spline_shape[axis])
            i = np.floor(t)
            w = _bspline_weights_zone(np.asarray([t - i], dtype=real_dtype), order)[0]  # (order+1,)
            idx = (int(i) + offsets) % spline_shape[axis]
            gathered = np.take(acc, idx, axis=1)  # (n_band, order+1, ...)
            acc = np.einsum("nk...,k->n...", gathered, w.astype(coeffs.dtype))
        out[:, p] = np.real(acc)
    return out


def _build_bspline_leaf(structure: Structure_data, order: int = _BSPLINE_ORDER) -> AOs_bspline_data:
    """Interpolating B-spline leaf reproducing two smooth periodic bands.

    Samples the bands on the spline grid, prefilters them into interpolating
    coefficients (``bspline_prefilter_periodic``) and builds the leaf. Fully
    deterministic, so every cross-check is reproducible.
    """
    ns = _BSPLINE_NS
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    frac = np.stack([fx, fy, fz], axis=-1)  # (N0, N1, N2, 3)
    vals = _bspline_band_functions(frac)  # (n_band, N0, N1, N2)
    coeffs_cube = bspline_prefilter_periodic(vals.astype(np.complex128), order)
    coeffs_flat = coeffs_cube.reshape(coeffs_cube.shape[0], -1)
    return AOs_bspline_data.build(structure, coeffs_flat, ns, order)


def _setup_grad_lap_test(
    ao_kind: str,
    periodic: bool,
    seed: int,
) -> tuple[AOs_cart_gto_data | AOs_sphe_gto_data, np.ndarray]:
    """Build ``(aos_data, r_carts)`` for the grad / Laplacian cross-checks.

    The 12 grad/lap tests below all need the same scaffolding (build a small
    basis, draw electron positions) and differ only in which production /
    oracle pair they cross-check. Centralising the setup keeps each test
    function focused on the comparison.

    Uniform ``K = 1`` throughout -- contraction is covered separately by
    :py:func:`test_compute_AOs_public_vs_debug_molecular`; the grad/lap
    tests focus on derivative correctness across the cart / sphe dispatch
    shims and the molecular / PBC fast-path branches.

    Molecular: ``l_max = 2`` (s + p + d, 9 sphe / 10 cart AOs) with one atom
    per AO at random positions in ``[-0.5, 0.5]`` Bohr; electrons in
    ``[-1.5, 1.5]`` Bohr.

    Periodic: cubic 10 Bohr cell with a single hydrogen at ``(5, 5, 5)``.
    ``l_max = 1`` (s + full p, 4 molecular AOs) so the expanded ``num_ao
    = 4 * 275 = 1100`` keeps the kernel-debug ``shell_indices`` construction
    sub-second. Electrons inside the cell at random positions in
    ``[2.0, 8.0]`` Bohr (well away from the cell wrap boundaries so the
    finite-difference stencil stays single-image-dominated).

    Non-GTO leaves (``ao_kind in {"pw", "bspline"}``) are periodic-only; they
    ignore the ``periodic`` flag (always PBC) and route through the shared
    builders :py:func:`_build_pw_leaf` / :py:func:`_build_bspline_leaf` on a
    small cubic cell, with electrons drawn well inside the cell.
    """
    rng = np.random.default_rng(seed)
    if ao_kind in ("pw", "bspline"):
        a = _NONGTO_A
        structure = _nongto_cubic_structure(a)
        if ao_kind == "pw":
            aos_data = _build_pw_leaf(structure)
        else:
            aos_data = _build_bspline_leaf(structure)
        aos_data.sanity_check()
        r_carts = rng.uniform(0.5, a - 0.5, size=(5, 3)).astype(np.float64)
        return aos_data, r_carts
    if periodic:
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos_pbc_mol = _make_aos_data(structure_pbc, ao_kind, l_max=1, uniform=True)
        aos_data = _apply_image_expansion(
            aos_pbc_mol,
            structure_pbc,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        r_carts = rng.uniform(2.0, 8.0, size=(3, 3)).astype(np.float64)
    else:
        if ao_kind == "sphe":
            num_ao = sum(2 * l + 1 for l in range(3))  # 9
        else:
            num_ao = sum((l + 1) * (l + 2) // 2 for l in range(3))  # 10
        positions = rng.uniform(-0.5, 0.5, size=(num_ao, 3)).astype(np.float64)
        structure_mol = Structure_data(
            pbc_flag=False,
            positions=positions,
            atomic_numbers=tuple([0] * num_ao),
            element_symbols=tuple(["X"] * num_ao),
            atomic_labels=tuple(["X"] * num_ao),
        )
        aos_data = _make_aos_data(structure_mol, ao_kind, l_max=2, uniform=True)
        r_carts = rng.uniform(-1.5, 1.5, size=(5, 3)).astype(np.float64)
    aos_data.sanity_check()
    return aos_data, r_carts


# ---------------------------------------------------------------------------
# Gradient cross-checks.
# Each pair (analytic vs autodiff, autodiff vs FD, analytic vs FD) is run on
# both molecular and periodic basis instances and over both AO kinds.
# ---------------------------------------------------------------------------


class TestComputeAOsGradLaplacian:
    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_AOs_grad_analytic_vs_auto_molecular(self, ao_kind: str):
        """Check full analytical/AD correctness and mixed/full fidelity.

        Autodiff goes through ``compute_AOs``. It remains the strict
        full-precision correctness oracle; mixed compares the analytical
        production route with an independently rebuilt full case.
        """

        def case_factory():
            return _setup_grad_lap_test(ao_kind, periodic=False, seed=2025)

        def analytical(case):
            aos_data, r_carts = case
            return compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        aos_data, r_carts = case_factory()
        gx_a, gy_a, gz_a = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)
        gx_u, gy_u, gz_u = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        for name, an, au in [("gx", gx_a, gx_u), ("gy", gy_a, gy_u), ("gz", gz_a, gz_u)]:
            assert not np.any(np.isnan(np.asarray(an))), f"NaN in analytic {name} ({ao_kind}, molecular)"
            assert not np.any(np.isnan(np.asarray(au))), f"NaN in autodiff {name} ({ao_kind}, molecular)"
            np.testing.assert_allclose(an, au, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe", "pw", "bspline"])
    def test_AOs_grad_analytic_vs_auto_periodic(self, ao_kind: str):
        """PBC sibling of :py:func:`test_AOs_grad_analytic_vs_auto_molecular`.

        Exercises both the kernel-level grad analytics and the public-wrapper
        post-kernel image reduction on the expanded ``aos_data``. autodiff
        goes through ``compute_AOs`` (already reshape-summed for PBC), so the
        two paths must agree strictly in full precision. Mixed instead checks
        the analytical route against an independently rebuilt full case at the
        public-API rank ``(num_ao_molecular, N_e)``.
        """

        def case_factory():
            return _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)

        def analytical(case):
            aos_data, r_carts = case
            return compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        aos_data, r_carts = case_factory()
        gx_a, gy_a, gz_a = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)
        gx_u, gy_u, gz_u = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        for name, an, au in [("gx", gx_a, gx_u), ("gy", gy_a, gy_u), ("gz", gz_a, gz_u)]:
            assert not np.any(np.isnan(np.asarray(an))), f"NaN in analytic {name} ({ao_kind}, periodic)"
            assert not np.any(np.isnan(np.asarray(au))), f"NaN in autodiff {name} ({ao_kind}, periodic)"
            np.testing.assert_allclose(an, au, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_AOs_grad_auto_vs_numerical_molecular(self, ao_kind: str):
        """``_compute_AOs_grad_autodiff`` matches the 1st-order central FD
        oracle ``_compute_AOs_grad_debug`` on a molecular instance.
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=False, seed=2025)
        gx_u, gy_u, gz_u = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)
        gx_n, gy_n, gz_n = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        for name, au, fd in [("gx", gx_u, gx_n), ("gy", gy_u, gy_n), ("gz", gz_u, gz_n)]:
            assert not np.any(np.isnan(np.asarray(au))), f"NaN in autodiff {name} ({ao_kind}, molecular)"
            assert not np.any(np.isnan(np.asarray(fd))), f"NaN in FD {name} ({ao_kind}, molecular)"
            np.testing.assert_allclose(au, fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-ao-grad-molecular",
            pytest_node_id="tests/test_jqmc_kernel/test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_grad_auto_vs_numerical_molecular",
            fixture_parameters=fd_case_parameters("fd-ad-ao-grad-molecular", ao_kind=ao_kind),
            coordinates=(r_carts, np.empty((0, 3))),
            coordinate_generator="tests.test_jqmc_kernel.test_atomic_orbital._setup_grad_lap_test",
            coordinate_seed=2025,
            compared_pairs=((gx_u, gx_n), (gy_u, gy_n), (gz_u, gz_n)),
            atol=atol,
            rtol=rtol,
            zones=("ao_eval", "ao_grad_lap"),
            level="strict",
            stencil="central-first-order-2",
            step=1.0e-5,
            direction="cartesian-basis",
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe", "pw", "bspline"])
    def test_AOs_grad_auto_vs_numerical_periodic(self, ao_kind: str):
        """PBC sibling of :py:func:`test_AOs_grad_auto_vs_numerical_molecular`.

        FD on the expanded ``compute_AOs`` and autodiff over the same wrapper
        both produce ``(num_ao_molecular, N_e)`` arrays; agreement confirms
        the image-axis reduction commutes with both differentiation paths.
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)
        gx_u, gy_u, gz_u = _compute_AOs_grad_autodiff(aos_data=aos_data, r_carts=r_carts)
        gx_n, gy_n, gz_n = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        for name, au, fd in [("gx", gx_u, gx_n), ("gy", gy_u, gy_n), ("gz", gz_u, gz_n)]:
            assert not np.any(np.isnan(np.asarray(au))), f"NaN in autodiff {name} ({ao_kind}, periodic)"
            assert not np.any(np.isnan(np.asarray(fd))), f"NaN in FD {name} ({ao_kind}, periodic)"
            np.testing.assert_allclose(au, fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-ao-grad-periodic",
            pytest_node_id="tests/test_jqmc_kernel/test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_grad_auto_vs_numerical_periodic",
            fixture_parameters=fd_case_parameters("fd-ad-ao-grad-periodic", ao_kind=ao_kind),
            coordinates=(r_carts, np.empty((0, 3))),
            coordinate_generator="tests.test_jqmc_kernel.test_atomic_orbital._setup_grad_lap_test",
            coordinate_seed=2025,
            compared_pairs=((gx_u, gx_n), (gy_u, gy_n), (gz_u, gz_n)),
            atol=atol,
            rtol=rtol,
            zones=("ao_eval", "ao_grad_lap"),
            level="strict",
            stencil="central-first-order-2",
            step=1.0e-5,
            direction="cartesian-basis",
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_AOs_grad_analytic_vs_numerical_molecular(self, ao_kind: str):
        """Analytic ``compute_AOs_grad`` matches FD ``_compute_AOs_grad_debug``
        on a molecular instance.
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=False, seed=2025)
        gx_a, gy_a, gz_a = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)
        gx_n, gy_n, gz_n = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        for name, an, fd in [("gx", gx_a, gx_n), ("gy", gy_a, gy_n), ("gz", gz_a, gz_n)]:
            assert not np.any(np.isnan(np.asarray(an))), f"NaN in analytic {name} ({ao_kind}, molecular)"
            assert not np.any(np.isnan(np.asarray(fd))), f"NaN in FD {name} ({ao_kind}, molecular)"
            np.testing.assert_allclose(an, fd, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe", "pw", "bspline"])
    def test_AOs_grad_analytic_vs_numerical_periodic(self, ao_kind: str):
        """PBC sibling of :py:func:`test_AOs_grad_analytic_vs_numerical_molecular`."""
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)
        gx_a, gy_a, gz_a = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)
        gx_n, gy_n, gz_n = _compute_AOs_grad_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        for name, an, fd in [("gx", gx_a, gx_n), ("gy", gy_a, gy_n), ("gz", gz_a, gz_n)]:
            assert not np.any(np.isnan(np.asarray(an))), f"NaN in analytic {name} ({ao_kind}, periodic)"
            assert not np.any(np.isnan(np.asarray(fd))), f"NaN in FD {name} ({ao_kind}, periodic)"
            np.testing.assert_allclose(an, fd, atol=atol, rtol=rtol)
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # Laplacian cross-checks.
    # Same three pairings as the gradient tests. ``analytic_vs_numerical`` and
    # ``auto_vs_numerical`` use the medium tolerance level because the 4th-order
    # central FD truncation error sets the floor (well above ao_eval round-off).
    # ---------------------------------------------------------------------------

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_AOs_laplacian_analytic_vs_auto_molecular(self, ao_kind: str):
        """Check full analytical/AD correctness and mixed/full fidelity.

        The full oracle is the Hessian-diagonal trace through ``compute_AOs``;
        mixed compares the analytical Laplacian with its independently rebuilt
        full-precision production twin.
        """

        def case_factory():
            return _setup_grad_lap_test(ao_kind, periodic=False, seed=2025)

        def analytical(case):
            aos_data, r_carts = case
            return compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        aos_data, r_carts = case_factory()
        lap_a = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)
        lap_u = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(lap_a))), f"NaN in analytic ({ao_kind}, molecular)"
        assert not np.any(np.isnan(np.asarray(lap_u))), f"NaN in autodiff ({ao_kind}, molecular)"
        np.testing.assert_allclose(lap_a, lap_u, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe", "pw", "bspline"])
    def test_AOs_laplacian_analytic_vs_auto_periodic(self, ao_kind: str):
        """PBC sibling of :py:func:`test_AOs_laplacian_analytic_vs_auto_molecular`."""

        def case_factory():
            return _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)

        def analytical(case):
            aos_data, r_carts = case
            return compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        aos_data, r_carts = case_factory()
        lap_a = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)
        lap_u = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(lap_a))), f"NaN in analytic ({ao_kind}, periodic)"
        assert not np.any(np.isnan(np.asarray(lap_u))), f"NaN in autodiff ({ao_kind}, periodic)"
        np.testing.assert_allclose(lap_a, lap_u, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_AOs_laplacian_analytic_vs_numerical_molecular(self, ao_kind: str):
        """Analytic ``compute_AOs_laplacian`` matches the 4th-order central FD
        oracle ``_compute_AOs_laplacian_debug`` on a molecular instance.

        The 4th-order FD with ``h = 1e-3`` has truncation error well above
        ``ao_eval`` round-off; medium tolerance is required.
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=False, seed=2025)
        lap_a = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)
        lap_n = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "medium")
        assert not np.any(np.isnan(np.asarray(lap_a))), f"NaN in analytic ({ao_kind}, molecular)"
        assert not np.any(np.isnan(np.asarray(lap_n))), f"NaN in FD ({ao_kind}, molecular)"
        np.testing.assert_allclose(lap_a, lap_n, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe", "pw", "bspline"])
    def test_AOs_laplacian_analytic_vs_numerical_periodic(self, ao_kind: str):
        """PBC sibling of :py:func:`test_AOs_laplacian_analytic_vs_numerical_molecular`.

        Also exercises the molecular-rank ``shape`` check inside
        ``_compute_AOs_laplacian_debug`` (which consults ``num_ao``, the
        molecular rank the per-leaf methods reduce the kernel output to).
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)
        lap_a = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)
        lap_n = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "medium")
        assert not np.any(np.isnan(np.asarray(lap_a))), f"NaN in analytic ({ao_kind}, periodic)"
        assert not np.any(np.isnan(np.asarray(lap_n))), f"NaN in FD ({ao_kind}, periodic)"
        np.testing.assert_allclose(lap_a, lap_n, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_AOs_laplacian_auto_vs_numerical_molecular(self, ao_kind: str):
        """``_compute_AOs_laplacian_autodiff`` matches FD
        ``_compute_AOs_laplacian_debug`` on a molecular instance.
        """
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=False, seed=2025)
        lap_u = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
        lap_n = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "medium")
        assert not np.any(np.isnan(np.asarray(lap_u))), f"NaN in autodiff ({ao_kind}, molecular)"
        assert not np.any(np.isnan(np.asarray(lap_n))), f"NaN in FD ({ao_kind}, molecular)"
        np.testing.assert_allclose(lap_u, lap_n, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-ao-laplacian-molecular",
            pytest_node_id="tests/test_jqmc_kernel/test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_laplacian_auto_vs_numerical_molecular",
            fixture_parameters=fd_case_parameters("fd-ad-ao-laplacian-molecular", ao_kind=ao_kind),
            coordinates=(r_carts, np.empty((0, 3))),
            coordinate_generator="tests.test_jqmc_kernel.test_atomic_orbital._setup_grad_lap_test",
            coordinate_seed=2025,
            compared_pairs=((lap_u, lap_n),),
            atol=atol,
            rtol=rtol,
            zones=("ao_eval", "ao_grad_lap"),
            level="medium",
            stencil="central-second-order-4",
            step=1.0e-3,
            direction="cartesian-basis",
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe", "pw", "bspline"])
    def test_AOs_laplacian_auto_vs_numerical_periodic(self, ao_kind: str):
        """PBC sibling of :py:func:`test_AOs_laplacian_auto_vs_numerical_molecular`."""
        aos_data, r_carts = _setup_grad_lap_test(ao_kind, periodic=True, seed=2025)
        lap_u = _compute_AOs_laplacian_autodiff(aos_data=aos_data, r_carts=r_carts)
        lap_n = _compute_AOs_laplacian_debug(aos_data=aos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap"], "medium")
        assert not np.any(np.isnan(np.asarray(lap_u))), f"NaN in autodiff ({ao_kind}, periodic)"
        assert not np.any(np.isnan(np.asarray(lap_n))), f"NaN in FD ({ao_kind}, periodic)"
        np.testing.assert_allclose(lap_u, lap_n, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-ao-laplacian-periodic",
            pytest_node_id="tests/test_jqmc_kernel/test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_laplacian_auto_vs_numerical_periodic",
            fixture_parameters=fd_case_parameters("fd-ad-ao-laplacian-periodic", ao_kind=ao_kind),
            coordinates=(r_carts, np.empty((0, 3))),
            coordinate_generator="tests.test_jqmc_kernel.test_atomic_orbital._setup_grad_lap_test",
            coordinate_seed=2025,
            compared_pairs=((lap_u, lap_n),),
            atol=atol,
            rtol=rtol,
            zones=("ao_eval", "ao_grad_lap"),
            level="medium",
            stencil="central-second-order-4",
            step=1.0e-3,
            direction="cartesian-basis",
        )
        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file",
        [
            "water_ccecp_ccpvqz.h5",  # spherical (l up to f)
            "H2_ae_ccpvdz_cart.h5",  # Cartesian
            "N_ae_ccpvdz_cart.h5",  # Cartesian, larger
        ],
    )
    @pytest.mark.light
    def test_AOs_value_grad_lap_molecular(self, trexio_file: str):
        """Fused ``compute_AOs_value_grad_lap`` matches the standalone APIs on
        molecular TREXIO instances.

        grad parity is at the ao_eval zone tolerance (not bitwise): the fused
        kernel mirrors the standalone grad kernels' shared body, but on CPU the
        fused path is the sphe-VGL FFI/C kernel which differs from the
        standalone XLA grad by ~1 ULP (accepted per the project tolerance
        policy). lap is allowed up to ULP-level differences: the
        standalone ``_compute_AOs_laplacian_*`` and the fused
        ``_compute_AOs_value_grad_lap_*`` share the same source expression but
        XLA may reorder the upstream FMA chain that produces ``lap_dup``
        differently between the two ``@jit`` boundaries (the per-primitive
        reduction layer is identical). value parity is bounded by the
        multiplication ordering between the fused kernel (which reuses the
        grad/lap ``phi``) and the standalone ``compute_AOs`` (which builds the
        polynomial separately).
        """
        (
            _structure,
            aos_data,
            *_rest,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        aos_data.sanity_check()
        assert aos_data._n_image == 1, "molecular TREXIO fixture must yield _n_image == 1"

        rng = np.random.default_rng(20260430)
        r_carts = (rng.standard_normal((6, 3)) * 1.5).astype(np.float64)

        val_f, gx_f, gy_f, gz_f, lap_f = compute_AOs_value_grad_lap(aos_data=aos_data, r_carts=r_carts)
        val_s = compute_AOs(aos_data=aos_data, r_carts=r_carts)
        gx_s, gy_s, gz_s = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)
        lap_s = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

        # NaN guards.
        for arr in (val_f, gx_f, gy_f, gz_f, lap_f, val_s, gx_s, gy_s, gz_s, lap_s):
            assert not np.any(np.isnan(np.asarray(arr)))

        # grad: ao_eval-zone tolerance (matches the periodic sibling and the
        # lap/value checks below). The fused and standalone grad kernels share
        # the same closed-form expression, but on CPU the fused
        # ``compute_AOs_value_grad_lap`` is routed to the sphe-VGL FFI/C kernel
        # (``jqmc_ao_sphe_vgl``), which differs from the standalone XLA grad by
        # ~1 ULP (different FMA/exp ordering). ULP-level FFI-vs-XLA differences
        # are accepted (project tolerance policy); this is a fused-vs-standalone
        # consistency check between two PRODUCTION paths, not an oracle check.
        atol_grad, rtol_grad = get_tolerance("ao_eval", "strict")
        assert_allclose(np.asarray(gx_f), np.asarray(gx_s), atol=atol_grad, rtol=rtol_grad)
        assert_allclose(np.asarray(gy_f), np.asarray(gy_s), atol=atol_grad, rtol=rtol_grad)
        assert_allclose(np.asarray(gz_f), np.asarray(gz_s), atol=atol_grad, rtol=rtol_grad)

        # lap: ao_eval-zone tolerance. The standalone and fused laplacian
        # kernels evaluate the same closed-form expression but live in
        # different ``@jit`` boundaries, so XLA may reassociate the upstream
        # FMA chain producing ``lap_dup``; strictly ULP-level differences
        # are expected and tolerated.
        atol_lap, rtol_lap = get_tolerance("ao_eval", "strict")
        assert_allclose(np.asarray(lap_f), np.asarray(lap_s), atol=atol_lap, rtol=rtol_lap)

        # value: tight ao_eval-zone tolerance. Fused reuses the grad/lap
        # ``phi`` (left-to-right multiplication chain), while standalone
        # ``compute_AOs`` parenthesises the polynomial separately -- strictly
        # ULP-level differences are allowed.
        atol_val, rtol_val = get_tolerance("ao_eval", "strict")
        assert_allclose(np.asarray(val_f), np.asarray(val_s), atol=atol_val, rtol=rtol_val)

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file",
        [
            "lithium.h5",  # PBC; auto-expanded by ``read_trexio_file``
        ],
    )
    @pytest.mark.light
    def test_AOs_value_grad_lap_periodic(self, trexio_file: str):
        """PBC sibling of :py:func:`test_AOs_value_grad_lap_molecular`.

        The TREXIO loader auto-applies ``_apply_image_expansion`` to
        periodic AOs, so ``aos_data._n_image > 1`` on entry. All four
        invocations below go through the public AO wrappers; the
        post-kernel reshape-sum is applied to all five fused outputs as
        well as the three standalone outputs. Unlike the molecular
        sibling, grad cannot be asserted at bit-exact tolerance because
        JAX may order the image-axis fp64 additions differently between
        the fused and standalone reduce sites, so grad / val / lap all
        compare at the ``ao_eval`` strict ULP level. Sampling ``r_carts``
        from a wide standard normal keeps the comparison free of
        cell-edge degenerate cases while still landing on points inside
        the cell after the implicit image contraction.
        """
        (
            _structure,
            aos_data,
            *_rest,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        aos_data.sanity_check()
        assert aos_data._n_image > 1, f"PBC TREXIO fixture {trexio_file!r} must yield _n_image > 1, got {aos_data._n_image}"

        rng = np.random.default_rng(20260430)
        r_carts = (rng.standard_normal((6, 3)) * 1.5).astype(np.float64)

        val_f, gx_f, gy_f, gz_f, lap_f = compute_AOs_value_grad_lap(aos_data=aos_data, r_carts=r_carts)
        val_s = compute_AOs(aos_data=aos_data, r_carts=r_carts)
        gx_s, gy_s, gz_s = compute_AOs_grad(aos_data=aos_data, r_carts=r_carts)
        lap_s = compute_AOs_laplacian(aos_data=aos_data, r_carts=r_carts)

        for arr in (val_f, gx_f, gy_f, gz_f, lap_f, val_s, gx_s, gy_s, gz_s, lap_s):
            assert not np.any(np.isnan(np.asarray(arr)))

        # Public-API shape contract: post-kernel reshape-sum collapses the
        # image axis on every output -> ``(num_ao, N_e)``.
        expected_shape = (aos_data.num_ao, r_carts.shape[0])
        for arr in (val_f, gx_f, gy_f, gz_f, lap_f, val_s, gx_s, gy_s, gz_s, lap_s):
            assert np.asarray(arr).shape == expected_shape

        # Periodic case: the post-kernel reshape-sum collapses the image
        # axis by summing fp64 contributions in a JAX-determined order.
        # Fused and standalone paths both reduce by the same numpy
        # expression, but JAX may schedule the underlying additions
        # differently between the two call sites, so fp64 non-associativity
        # produces noise at the ``ao_eval`` strict level even for grad
        # (~1.5e-13 observed). The molecular sibling stays bit-exact
        # because there is no image axis to reduce over.
        atol_grad, rtol_grad = get_tolerance("ao_eval", "strict")
        assert_allclose(np.asarray(gx_f), np.asarray(gx_s), atol=atol_grad, rtol=rtol_grad)
        assert_allclose(np.asarray(gy_f), np.asarray(gy_s), atol=atol_grad, rtol=rtol_grad)
        assert_allclose(np.asarray(gz_f), np.asarray(gz_s), atol=atol_grad, rtol=rtol_grad)

        atol_lap, rtol_lap = get_tolerance("ao_eval", "strict")
        assert_allclose(np.asarray(lap_f), np.asarray(lap_s), atol=atol_lap, rtol=rtol_lap)

        atol_val, rtol_val = get_tolerance("ao_eval", "strict")
        assert_allclose(np.asarray(val_f), np.asarray(val_s), atol=atol_val, rtol=rtol_val)

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file",
        [
            "water_ccecp_ccpvqz.h5",
            "H2_ae_ccpvdz_cart.h5",
        ],
    )
    @pytest.mark.light
    def test_fused_AOs_dtypes_match_zones_molecular(self, trexio_file: str):
        """``compute_AOs_value_grad_lap`` outputs are pinned to their zones on
        molecular instances.

        val <-> ``ao_eval`` (fp32 mixed / fp64 full); gx/gy/gz/lap <->
        ``ao_grad_lap`` (fp64 always).
        """
        (
            _structure,
            aos_data,
            *_rest,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        aos_data.sanity_check()
        assert aos_data._n_image == 1, "molecular TREXIO fixture must yield _n_image == 1"

        rng = np.random.default_rng(20260430)
        r_carts = (rng.standard_normal((4, 3)) * 1.5).astype(np.float64)
        val_f, gx_f, gy_f, gz_f, lap_f = compute_AOs_value_grad_lap(aos_data=aos_data, r_carts=r_carts)

        eval_dtype = get_dtype_jnp("ao_eval")
        gradlap_dtype = get_dtype_jnp("ao_grad_lap")
        assert val_f.dtype == eval_dtype, f"val.dtype = {val_f.dtype}, expected {eval_dtype}"
        assert gx_f.dtype == gradlap_dtype, f"gx.dtype = {gx_f.dtype}, expected {gradlap_dtype}"
        assert gy_f.dtype == gradlap_dtype, f"gy.dtype = {gy_f.dtype}, expected {gradlap_dtype}"
        assert gz_f.dtype == gradlap_dtype, f"gz.dtype = {gz_f.dtype}, expected {gradlap_dtype}"
        assert lap_f.dtype == gradlap_dtype, f"lap.dtype = {lap_f.dtype}, expected {gradlap_dtype}"

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file",
        [
            "lithium.h5",  # PBC; auto-expanded by ``read_trexio_file``
        ],
    )
    @pytest.mark.light
    def test_fused_AOs_dtypes_match_zones_periodic(self, trexio_file: str):
        """PBC sibling of :py:func:`test_fused_AOs_dtypes_match_zones_molecular`.

        The dtype contract is independent of ``pbc_flag`` -- the post-kernel
        image reduction is an ``np.sum`` along the image axis which preserves
        dtype, so the per-output zones (``ao_eval`` for ``val``,
        ``ao_grad_lap`` for grad / lap) are unchanged by PBC expansion.
        """
        (
            _structure,
            aos_data,
            *_rest,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        aos_data.sanity_check()
        assert aos_data._n_image > 1, f"PBC TREXIO fixture {trexio_file!r} must yield _n_image > 1, got {aos_data._n_image}"

        rng = np.random.default_rng(20260430)
        r_carts = (rng.standard_normal((4, 3)) * 1.5).astype(np.float64)
        val_f, gx_f, gy_f, gz_f, lap_f = compute_AOs_value_grad_lap(aos_data=aos_data, r_carts=r_carts)

        eval_dtype = get_dtype_jnp("ao_eval")
        gradlap_dtype = get_dtype_jnp("ao_grad_lap")
        assert val_f.dtype == eval_dtype, f"val.dtype = {val_f.dtype}, expected {eval_dtype}"
        assert gx_f.dtype == gradlap_dtype, f"gx.dtype = {gx_f.dtype}, expected {gradlap_dtype}"
        assert gy_f.dtype == gradlap_dtype, f"gy.dtype = {gy_f.dtype}, expected {gradlap_dtype}"
        assert gz_f.dtype == gradlap_dtype, f"gz.dtype = {gz_f.dtype}, expected {gradlap_dtype}"
        assert lap_f.dtype == gradlap_dtype, f"lap.dtype = {lap_f.dtype}, expected {gradlap_dtype}"

        jax.clear_caches()


class TestOverlapMatrix:
    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_overlap_matrix_analytic_vs_numerical_debug_molecular(self, ao_kind: str):
        """Analytic ``compute_overlap_matrix`` matches the numerical 3D-midpoint
        ``_compute_overlap_matrix_debug`` on a small molecular 2-atom basis.

        Cartesian and spherical bases are covered by the ao_kind parametrisation.
        """
        centers = np.array([[-0.45, 0.0, 0.0], [0.45, 0.0, 0.0]], dtype=np.float64)
        structure_data = Structure_data(
            pbc_flag=False,
            positions=centers,
            atomic_numbers=(0, 0),
            element_symbols=("X", "X"),
            atomic_labels=("X1", "X2"),
        )
        structure_data.sanity_check()

        common = dict(
            structure_data=structure_data,
            nucleus_index=(0, 1),
            num_ao=2,
            num_ao_prim=2,
            orbital_indices=(0, 1),
            exponents=np.array([1.20, 1.20], dtype=np.float64),
            coefficients=np.array([1.0, 1.0], dtype=np.float64),
            angular_momentums=(0, 0),
        )
        if ao_kind == "cart":
            aos_data = AOs_cart_gto_data._from_full_basis(
                **common,
                polynominal_order_x=(0, 0),
                polynominal_order_y=(0, 0),
                polynominal_order_z=(0, 0),
            )
        else:
            aos_data = AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=(0, 0))
        aos_data.sanity_check()

        overlap_an = np.asarray(compute_overlap_matrix(aos_data=aos_data), dtype=np.float64)
        overlap_num = _compute_overlap_matrix_debug(
            aos_data=aos_data,
            num_grid_points=41,
            tail_tolerance=1.0e-12,
        )

        atol, rtol = get_tolerance_min(("ao_eval",), "strict")
        assert not np.any(np.isnan(overlap_an)), f"NaN in analytic ({ao_kind}, molecular)"
        assert not np.any(np.isnan(overlap_num)), f"NaN in numerical ({ao_kind}, molecular)"
        np.testing.assert_allclose(overlap_an, overlap_num, atol=atol, rtol=rtol)
        np.testing.assert_allclose(overlap_an, overlap_an.T, atol=atol, rtol=rtol)
        assert np.all(np.diag(overlap_an) > 0.0)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_overlap_matrix_analytic_vs_numerical_debug_periodic(self, ao_kind: str):
        """PBC sibling of
        :py:func:`test_overlap_matrix_analytic_vs_numerical_debug_molecular`.

        Cross-checks two independent paths to the cell-domain periodic AO
        overlap

        .. math::
           S^{PBC}_{\\mu\\nu} \\;=\\; \\int_\\Omega
           \\overline{\\tilde\\chi_\\mu(r)}\\,\\tilde\\chi_\\nu(r)\\,dr

        on the helper-built fixture (single atom in cubic 10 Bohr,
        ``l_max=1`` so ``(alpha, L)`` keys ``{(1.0, 0), (1.0, 1)}`` drive
        ``_n_image == 275``):

          * ``compute_overlap_matrix`` (production): closed-form primitive
            overlaps summed over an overlap-owned lattice image set.
          * ``_compute_overlap_matrix_debug`` (numerical): 3D-midpoint
            quadrature on the parallelepiped cell, evaluating the periodic
            :math:`\\tilde\\chi_\\mu` via :py:func:`_compute_AOs_debug`
            (which is itself reshape-sum-reduced).

        The midpoint rule on a 10 Bohr cell with ``num_grid_points = 25``
        has truncation error ``O((h / 24) * f'') ~= 1e-3`` for the
        ``alpha = 1.0`` Gaussian product near the cell centre -- well
        within the ``medium`` tolerance level. Tighter tolerance would
        require ``num_grid_points >> 25`` which would explode the
        ``num_ao = 1100``-rank kernel-debug runtime.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos_pbc_mol = _make_aos_data(structure_pbc, ao_kind, l_max=1, uniform=True)
        aos_data = _apply_image_expansion(
            aos_pbc_mol,
            structure_pbc,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        assert aos_data._n_image > 1, "fixture must exercise the PBC fold path"
        M_c = aos_data.num_ao

        overlap_an = np.asarray(compute_overlap_matrix(aos_data=aos_data), dtype=np.float64)
        overlap_num = _compute_overlap_matrix_debug(
            aos_data=aos_data,
            num_grid_points=25,
        )

        # Shape contract: PBC overlap returns molecular rank, not kernel rank.
        assert overlap_an.shape == (M_c, M_c)
        assert overlap_num.shape == (M_c, M_c)

        # medium tolerance: midpoint quadrature truncation dominates.
        atol, rtol = get_tolerance_min(("ao_eval",), "medium")
        assert not np.any(np.isnan(overlap_an)), f"NaN in analytic ({ao_kind}, periodic)"
        assert not np.any(np.isnan(overlap_num)), f"NaN in numerical ({ao_kind}, periodic)"
        np.testing.assert_allclose(overlap_an, overlap_num, atol=atol, rtol=rtol)
        # Symmetry and positive-diagonal hold exactly (no FD error here).
        atol_s, rtol_s = get_tolerance_min(("ao_eval",), "strict")
        np.testing.assert_allclose(overlap_an, overlap_an.T, atol=atol_s, rtol=rtol_s)
        assert np.all(np.diag(overlap_an) > 0.0)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_overlap_matrix_multi_bucket_periodic(self, ao_kind: str):
        """Per-shell ragged overlap with MIXED exponents -> MULTIPLE multiplicity
        buckets, the case the single-exponent fixtures miss. A diffuse shell
        keeps many images, a tight shell few, so the home AOs are NOT the first
        ``M_c`` kernel-AOs; the analytic overlap must still match the numerical
        integral, be Hermitian, and be POSITIVE-DEFINITE (real and complex
        Bloch). Regression for the bucket-major ``T_0`` home-row recovery in
        ``_compute_overlap_matrix_cart_analytic``.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        structure = make_pbc_structure(
            cell,
            positions=np.array([[1.0, 1.2, 0.8]], dtype=np.float64),
            atomic_numbers=(3,),
            element_symbols=("Li",),
            atomic_labels=("Li1",),
        )
        # s (diffuse), p (medium), s (tight) -> per-shell image counts differ.
        exps = np.array([0.15, 0.4, 0.4, 0.4, 3.0], dtype=np.float64)
        ang = (0, 1, 1, 1, 0)
        common = dict(
            structure_data=structure,
            nucleus_index=(0, 0, 0, 0, 0),
            num_ao=5,
            num_ao_prim=5,
            orbital_indices=(0, 1, 2, 3, 4),
            exponents=exps,
            coefficients=np.ones(5),
            angular_momentums=ang,
        )
        if ao_kind == "sphe":
            base = AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=(0, -1, 0, 1, 0))
        else:
            base = AOs_cart_gto_data._from_full_basis(
                **common,
                polynominal_order_x=(0, 1, 0, 0, 0),
                polynominal_order_y=(0, 0, 1, 0, 0),
                polynominal_order_z=(0, 0, 0, 1, 0),
            )
        leaf = _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
        assert len({int(b[0]) for b in leaf._ragged_buckets}) >= 2, "fixture must produce multiple multiplicity buckets"

        atol, rtol = get_tolerance("ao_eval", "medium")
        for cleaf, is_cplx in ((leaf, False), (_promote_pgto_to_complex(leaf, (0.27, -0.1, 0.13)), True)):
            comparison_iteration = str(is_cplx)  # noqa: F841 - captured by the GDD iteration receipt
            S = np.asarray(compute_overlap_matrix(aos_data=cleaf))
            assert S.shape == (5, 5)
            np.testing.assert_allclose(S, S.conj().T, atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_BIT_EXACT)
            # Positive-definiteness is the load-bearing guard: the bucket-major
            # bug produced negative eigenvalues (an invalid Gram matrix), which
            # blew up the S-metric polar retraction in basis optimisation.
            ev = np.linalg.eigvalsh(0.5 * (S + S.conj().T))
            assert ev.min() > 0.0, f"overlap must be positive-definite ({ao_kind}, complex={is_cplx}); eig_min={ev.min()}"
            # Independent, unchunked numerical cross-check.  In particular the
            # complex case guards the Hermitian Gram convention chi* chi^T;
            # a transpose-only implementation is generally neither equal to
            # this oracle nor positive definite at a non-Gamma twist.
            S_num = _compute_overlap_matrix_debug(aos_data=cleaf, num_grid_points=60)
            np.testing.assert_allclose(S, S_num, atol=max(atol, ATOL_AO_OVERLAP_GRID_QUADRATURE), rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium.h5", "lithium_k025.h5"])
    def test_periodic_lattice_overlap_has_its_own_converged_primitive_pair_cutoff(self, trexio_file: str):
        """Regression for reusing the faster AO-value image cutoff in S(k)."""
        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        geminal = read_trexio_file(fixture, store_tuple=True)[4]
        orbitals = geminal.orb_data_up_spin
        aos_data = orbitals.aos_data
        target = get_pgto_lattice_overlap_term_tolerance(get_dtype_jnp("det_eval"))

        overlap = np.asarray(compute_overlap_matrix(aos_data))
        tighter = _compute_overlap_matrix_periodic_pgto_lattice(aos_data, term_tolerance=target * 1.0e-2)
        relative_change = np.linalg.norm(overlap - tighter, ord=np.inf) / np.linalg.norm(tighter, ord=np.inf)
        molecular = _detile_ragged_to_molecular(aos_data)
        loose = _apply_image_expansion(
            molecular,
            molecular.structure_data,
            tol=float(np.sqrt(AO_IMAGE_TRUNCATION_ATOL)),
        )
        if np.iscomplexobj(overlap):
            loose = _promote_pgto_to_complex(loose, aos_data.k_point)
        loose_overlap = np.asarray(compute_overlap_matrix(loose))

        np.testing.assert_allclose(overlap, overlap.conj().T, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_array_equal(overlap, loose_overlap)
        assert np.linalg.eigvalsh(overlap).min() > 0.0
        assert relative_change <= RTOL_PGTO_LATTICE_OVERLAP_CONVERGENCE_FACTOR * target
        coefficients = np.asarray(orbitals.mo_coefficients)
        gram = coefficients.conj() @ overlap @ coefficients.T
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("grid_size", [48])
    def test_overlap_matrix_plane_wave_analytic_vs_grid(self, grid_size):
        """``compute_overlap_matrix`` on the real cos/sin plane-wave leaf.

        The analytic cell-domain overlap (closed-form
        ``_compute_overlap_matrix_plane_wave``) is cross-checked against an
        independent fine fractional-midpoint grid integral over the cell,
        whose plane-wave values are evaluated in the ``ao_eval`` dtype and
        summed in the ``det_eval`` dtype. For the standard half-space basis (constant + cos/sin pairs over
        distinct reciprocal vectors) the matrix is diagonal: ``V`` for the
        constant and ``V / 2`` for every cos/sin function, ``V`` being the
        cell volume.
        """
        structure = _nongto_cubic_structure()
        aos = _build_pw_leaf(structure)
        aos.sanity_check()

        cell = np.asarray(structure.cell, dtype=np.float64)
        cell_volume = float(np.abs(np.linalg.det(cell)))

        overlap = np.asarray(compute_overlap_matrix(aos_data=aos), dtype=np.float64)
        assert overlap.shape == (aos.num_ao, aos.num_ao)

        # Expected diagonal: V for the constant (G = 0, cos), V / 2 otherwise.
        g2 = np.sum(np.asarray(aos.g_vectors) ** 2, axis=1)
        is_const = g2 <= 1.0e-24
        expected_diag = np.where(is_const, cell_volume, 0.5 * cell_volume)
        atol_s, rtol_s = get_tolerance_min(("ao_eval",), "strict")
        np.testing.assert_allclose(np.diag(overlap), expected_diag, atol=atol_s, rtol=rtol_s)
        # Off-diagonals vanish (orthogonal reciprocal-lattice basis).
        np.testing.assert_allclose(overlap - np.diag(np.diag(overlap)), np.zeros_like(overlap), atol=atol_s, rtol=rtol_s)
        np.testing.assert_allclose(overlap, overlap.T, atol=atol_s, rtol=rtol_s)

        # Independent brute-force grid integral over the cell. Precision: the PW
        # values are evaluated in the ``ao_eval`` dtype (the zone whose cast the
        # production closed form carries; see _pw_reference_values) and the
        # quadrature sum is accumulated in the ``det_eval`` dtype of the
        # public overlap wrapper.
        n = grid_size
        steps = (np.arange(n, dtype=np.float64) + 0.5) / n
        fa, fb, fc = np.meshgrid(steps, steps, steps, indexing="ij")
        r_frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1)
        r_carts = r_frac @ cell
        vals = _pw_reference_values(np.asarray(aos.g_vectors), np.asarray(aos.phase_is_sin), r_carts)  # (num_ao, N_g)
        assert np.all(np.isfinite(vals))
        vals = vals.astype(get_dtype_np("det_eval"))
        volume_element = cell_volume / float(n**3)
        overlap_grid = volume_element * (vals @ vals.T)
        atol, rtol = get_tolerance_min(("ao_eval",), "medium")
        np.testing.assert_allclose(overlap, overlap_grid, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("grid_size", [64])
    def test_overlap_matrix_bspline_vs_independent_grid(self, grid_size):
        """``compute_overlap_matrix`` on the periodic B-spline (band) leaf.

        The production overlap is the exact spline Gram matrix. It is
        cross-checked against an independent, finer fractional-midpoint grid
        integral evaluated through the public ``compute_AOs`` (medium, the
        grid's own quadrature error) and against the exact cell integral of
        the same AO values by ``order + 1`` Gauss-Legendre nodes on every knot
        interval (strict). The matrix is symmetric positive-definite (the two
        bands are linearly independent).
        """
        structure = _nongto_cubic_structure()
        aos = _build_bspline_leaf(structure)
        aos.sanity_check()

        overlap = np.asarray(compute_overlap_matrix(aos_data=aos), dtype=np.float64)
        assert overlap.shape == (aos.num_ao, aos.num_ao)
        atol_s, rtol_s = get_tolerance_min(("ao_eval",), "strict")
        np.testing.assert_allclose(overlap, overlap.T, atol=atol_s, rtol=rtol_s)
        eigvals = np.linalg.eigvalsh(overlap)
        assert np.all(eigvals > 0.0), f"overlap not positive-definite: eigvals={eigvals}"

        # Independent finer grid integral through the public evaluator.
        cell = np.asarray(structure.cell, dtype=np.float64)
        cell_volume = float(np.abs(np.linalg.det(cell)))
        n = grid_size
        steps = (np.arange(n, dtype=np.float64) + 0.5) / n
        fa, fb, fc = np.meshgrid(steps, steps, steps, indexing="ij")
        r_frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1)
        r_carts = r_frac @ cell
        vals = np.asarray(compute_AOs(aos, r_carts), dtype=np.float64)
        volume_element = cell_volume / float(n**3)
        overlap_ref = volume_element * (vals @ vals.T)

        atol, rtol = get_tolerance_min(("ao_eval",), "medium")
        np.testing.assert_allclose(overlap, overlap_ref, atol=atol, rtol=rtol)

        # Exact reference: the piecewise-polynomial integrand of degree
        # 2 * order per axis is integrated exactly by order + 1 Gauss-Legendre
        # nodes on every knot interval.
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(aos.spline_order + 1)
        axes = [((np.arange(k)[:, None] + 0.5 * (gl_nodes[None, :] + 1.0)) / k).ravel() for k in aos.spline_shape]
        axis_weights = [np.tile(0.5 * gl_weights, k) / k for k in aos.spline_shape]
        r_frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        weights = np.einsum("i,j,k->ijk", *axis_weights).ravel() * cell_volume
        vals = np.asarray(compute_AOs(aos, r_frac @ cell), dtype=np.float64)
        overlap_exact = (vals * weights[None, :]) @ vals.T
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(overlap, overlap_exact, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    def test_overlap_matrix_composite_periodic_and_molecular(self):
        """``compute_overlap_matrix`` supports a periodic composite, rejects a molecular one.

        A periodic composite (here ``plane-wave + B-spline``) has a well-defined
        cell-domain Gram matrix assembled from exact child and cross blocks.
        Its shape is ``(sum(child.num_ao), sum(child.num_ao))``; it is symmetric
        and finite, and its plane-wave diagonal block matches the analytic
        plane-wave overlap. These checks use the same input and the existing
        AO strict tolerance separately in full and mixed precision. No finite
        difference or numerical-quadrature reference is used here; matrix rank
        is not asserted.

        A molecular (open-boundary) GTO composite has no cell to integrate over,
        so it raises ``NotImplementedError``.
        """
        from jqmc.atomic_orbital import _compute_overlap_matrix_plane_wave

        structure = _nongto_cubic_structure()
        pw = _build_pw_leaf(structure)
        bspline = _build_bspline_leaf(structure)
        composite = pw + bspline
        composite.sanity_check()

        overlap = np.asarray(compute_overlap_matrix(aos_data=composite))
        n_total = pw.num_ao + bspline.num_ao
        assert overlap.shape == (n_total, n_total)
        assert np.all(np.isfinite(overlap))
        # Symmetry holds exactly (the kernel symmetrizes 0.5 * (S + S.T)).
        atol_s, rtol_s = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(overlap, overlap.T, atol=atol_s, rtol=rtol_s)

        # The (pw, pw) diagonal block matches the analytic plane-wave overlap.
        pw_block = overlap[: pw.num_ao, : pw.num_ao]
        pw_analytic = np.asarray(_compute_overlap_matrix_plane_wave(pw))
        np.testing.assert_allclose(pw_block, pw_analytic, atol=atol_s, rtol=rtol_s)

        # A molecular (open-boundary) GTO composite has no integration cell.
        structure_mol = Structure_data(
            pbc_flag=False,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        structure_mol.sanity_check()
        gto_a = _make_aos_data(structure_mol, "sphe", l_max=0, uniform=True)
        gto_b = _make_aos_data(structure_mol, "sphe", l_max=0, uniform=True)
        mol_composite = gto_a + gto_b
        mol_composite.sanity_check()
        with pytest.raises(NotImplementedError):
            compute_overlap_matrix(aos_data=mol_composite)

        jax.clear_caches()

    @pytest.mark.light
    def test_overlap_matrix_complex_composite_is_hermitian(self):
        """A non-Gamma composite retains the complex Hermitian AO Gram matrix."""
        structure = _nongto_cubic_structure()
        cell = np.asarray(structure.cell, dtype=np.float64)
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
        twist = (0.25, -0.125, 0.375)
        first = AOs_plane_wave_complex_data.build(
            structure,
            np.stack((np.zeros(3), reciprocal[0])),
            k_point=twist,
        )
        second = AOs_plane_wave_complex_data.build(
            structure,
            np.stack((reciprocal[1], reciprocal[2])),
            k_point=twist,
        )
        composite = first + second

        overlap = np.asarray(compute_overlap_matrix(composite))
        expected = abs(np.linalg.det(cell)) * np.eye(4, dtype=np.complex128)

        assert np.issubdtype(overlap.dtype, np.complexfloating)
        np.testing.assert_allclose(overlap, overlap.conj().T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(overlap, expected, atol=atol, rtol=rtol)

        jax.clear_caches()


# --- Composite (AOs_composite_data) tests (Step 11) ------------------------


class TestComposite:
    @pytest.mark.light
    def test_composite_add_operator_flattens_and_dispatches(self):
        """``a + b`` builds an ``AOs_composite_data`` whose ``compute_AOs`` /
        grad / lap / value_grad_lap concatenates the children's outputs
        along axis 0 (the AO axis). Associativity is enforced via
        construction-time flattening, so ``(a + b) + c == a + b + c`` ---
        the ``children`` tuple is always flat regardless of how the user
        parenthesises the sum.
        """
        from jqmc.atomic_orbital import AOs_composite_data

        cell = CELL_FIXTURES["cubic_10bohr"]
        structure_mol = Structure_data(
            pbc_flag=False,
            positions=np.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        cart = _make_aos_data(structure_mol, "cart", l_max=1, uniform=True)
        sphe = _make_aos_data(structure_mol, "sphe", l_max=1, uniform=True)
        cart.sanity_check()
        sphe.sanity_check()

        composite = cart + sphe
        assert isinstance(composite, AOs_composite_data)
        assert composite.num_ao == cart.num_ao + sphe.num_ao
        assert len(composite.children) == 2
        composite.sanity_check()

        rng = np.random.default_rng(20260603)
        r_carts = rng.uniform(-1.5, 1.5, size=(4, 3)).astype(np.float64)

        atol, rtol = get_tolerance("ao_eval", "strict")

        a = np.asarray(compute_AOs(cart, r_carts))
        b = np.asarray(compute_AOs(sphe, r_carts))
        c = np.asarray(compute_AOs(composite, r_carts))
        assert c.shape == (cart.num_ao + sphe.num_ao, r_carts.shape[0])
        np.testing.assert_allclose(c, np.concatenate([a, b], axis=0), atol=atol, rtol=rtol)

        # grad / lap / value_grad_lap follow the same concat pattern.
        gxa, gya, gza = compute_AOs_grad(cart, r_carts)
        gxb, gyb, gzb = compute_AOs_grad(sphe, r_carts)
        gxc, gyc, gzc = compute_AOs_grad(composite, r_carts)
        np.testing.assert_allclose(
            np.asarray(gxc), np.concatenate([np.asarray(gxa), np.asarray(gxb)], axis=0), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            np.asarray(gyc), np.concatenate([np.asarray(gya), np.asarray(gyb)], axis=0), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            np.asarray(gzc), np.concatenate([np.asarray(gza), np.asarray(gzb)], axis=0), atol=atol, rtol=rtol
        )

        lap_a = np.asarray(compute_AOs_laplacian(cart, r_carts))
        lap_b = np.asarray(compute_AOs_laplacian(sphe, r_carts))
        lap_c = np.asarray(compute_AOs_laplacian(composite, r_carts))
        np.testing.assert_allclose(lap_c, np.concatenate([lap_a, lap_b], axis=0), atol=atol, rtol=rtol)

        val_c, *_ = compute_AOs_value_grad_lap(composite, r_carts)
        val_ref = np.concatenate(
            [
                np.asarray(compute_AOs_value_grad_lap(cart, r_carts)[0]),
                np.asarray(compute_AOs_value_grad_lap(sphe, r_carts)[0]),
            ],
            axis=0,
        )
        np.testing.assert_allclose(np.asarray(val_c), val_ref, atol=atol, rtol=rtol)

        # Associativity: (a + b) + c has 3 flat children, not nested composite.
        chained = (cart + sphe) + cart
        assert isinstance(chained, AOs_composite_data)
        assert len(chained.children) == 3
        for ch in chained.children:
            assert not isinstance(ch, AOs_composite_data)

        jax.clear_caches()

    @pytest.mark.light
    def test_composite_structure_data_molecular_gto_plus_pgto(self):
        """The composite cross-child equality check uses ``structure_data``
        (the untiled molecular structure on every leaf, gto and pgto
        alike). ``_structure_data_equal`` is tolerance-aware so re-derived
        ``vec_a/b/c`` don't trip exact-equality.

        Also locks the open/periodic mixing rule: a molecular GTO leaf may
        NOT be combined with a periodic pGTO leaf, while a periodic
        ``(pgto + pgto)`` composite over the same molecule passes.
        """
        import pytest

        from jqmc.atomic_orbital import _structure_data_equal

        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        gto = _make_aos_data(structure_pbc, "sphe", l_max=1, uniform=True)
        # Build the pgto sibling on the same structure.
        pgto = _apply_image_expansion(gto, structure_pbc, tol=AO_IMAGE_TRUNCATION_ATOL)
        assert pgto._n_image > 1

        # The pgto leaf carries the MOLECULAR structure_data (untiled): the
        # image expansion is reconstructed in-trace from _image_translations, not
        # baked into structure_data.positions. So pgto.structure_data has the
        # molecular atom count and equals the gto leaf's structure_data.
        assert len(pgto.structure_data.atomic_numbers) == len(gto.structure_data.atomic_numbers)
        assert _structure_data_equal(pgto.structure_data, gto.structure_data)
        # The in-trace image-tiled centers expand to one per kernel-AO (M) under
        # the per-shell ragged layout: row k is R_{atom(k)} + T_{image(k)}.
        expanded_centers = np.asarray(pgto._expanded_positions_cart_np)
        assert expanded_centers.shape == (pgto._num_ao_kernel, 3)

        # Open/periodic mixing rule: a molecular GTO leaf may not be combined
        # with a periodic pGTO leaf.
        with pytest.raises(ValueError, match="open-boundary"):
            (gto + pgto).sanity_check()

        # A periodic (pgto + pgto) composite over the same molecule passes
        # sanity_check via the structure_data path: both leaves are
        # periodic and expose the same molecular structure_data.
        composite_pp = pgto + pgto
        composite_pp.sanity_check()
        assert composite_pp.num_ao == 2 * pgto.num_ao

        jax.clear_caches()


# --- pgto AO-count semantics ------------------------------------------------
# ``pgto.num_ao`` is the externally-observable molecular AO count, NOT the
# kernel-rank tiled count (which is ``_num_ao_kernel``).


class TestPGTOSemantics:
    @pytest.mark.light
    def test_pgto_num_ao_is_molecular_view(self):
        """``pgto.num_ao`` returns the molecular AO count, NOT the
        kernel-rank tiled count.

        ``pgto.num_ao == num_ao_molecular``;
        ``pgto._num_ao_kernel == n_image * num_ao_molecular`` is the
        kernel-rank count.

        This makes ``num_ao`` mean the same thing across gto and pgto
        leaves (the externally observable AO count), so downstream consumers
        (Slater determinant matrix dim, mo_coefficients column count,
        j_matrix square block) stop having to special-case pgto.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos_pbc_mol = _make_aos_data(structure_pbc, "sphe", l_max=1, uniform=True)
        M_c = aos_pbc_mol.num_ao
        pgto = _apply_image_expansion(aos_pbc_mol, structure_pbc, tol=AO_IMAGE_TRUNCATION_ATOL)

        assert pgto._n_image > 1
        # Contract: pgto.num_ao == molecular count.
        assert pgto.num_ao == M_c, (
            f"contract: pgto.num_ao must equal the molecular count {M_c}, "
            f"got {pgto.num_ao} (= n_image={pgto._n_image} * M_c={M_c})."
        )
        # Kernel rank is spelled _num_ao_kernel.
        assert pgto._num_ao_kernel == pgto._n_image * M_c, (
            f"contract: pgto._num_ao_kernel must equal kernel rank "
            f"({pgto._n_image} * {M_c} = {pgto._n_image * M_c}), got {pgto._num_ao_kernel}."
        )

        jax.clear_caches()

    @pytest.mark.light
    def test_pgto_j_matrix_size_uses_molecular_view(self):
        """When downstream code sizes a matrix by ``pgto.num_ao``
        (e.g. ``Jastrow_three_body_data.j_matrix`` of shape
        ``(num_ao, num_ao + 1)``), the resulting matrix is the molecular
        (not kernel-rank) shape.

        A kernel-rank ``num_ao`` would allocate an oversized
        ``(n_image * M_c, n_image * M_c + 1)`` j_matrix of which the consumer
        reads only the molecular sub-block.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos_pbc_mol = _make_aos_data(structure_pbc, "sphe", l_max=1, uniform=True)
        M_c = aos_pbc_mol.num_ao
        pgto = _apply_image_expansion(aos_pbc_mol, structure_pbc, tol=AO_IMAGE_TRUNCATION_ATOL)

        # The j_matrix sizing of the J3 builder, lifted out to make the
        # assertion direct:
        j_matrix_shape = (pgto.num_ao, pgto.num_ao + 1)
        assert j_matrix_shape == (M_c, M_c + 1), (
            f"the j_matrix sized by pgto.num_ao must be molecular shape ({M_c}, {M_c + 1}); got {j_matrix_shape}."
        )

        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # BCS / TR-pair convention invariant: ``aos_dn = aos_up.with_negated_k()``
    # must satisfy ``compute_AOs(aos_dn, r) == conj(compute_AOs(aos_up, r))`` so
    # the dn-channel orbital ``phi^{-k}(r) = conj(phi^k)(r)`` materialises
    # without spin-asymmetric conj at the kernel level. This pins the
    # ``-k`` AO leaf semantics that the BCS / TR-pair singlet representation
    # depends on across all loader / builder / kernel paths.
    #
    # Regression target: conjugating only the MO coefficients (leaving the +k AO
    # leaf in place for the dn channel) builds a mixed-phase orbital that is
    # *neither* ``phi^k`` nor ``phi^{-k}``. Such a leaf still type-checks through
    # the kernel and passes the cross-class consistency test
    # (``test_AGPn_SD_eval_pbc_complex``), but VMC of a loaded pyscf kRHF
    # wavefunction then disagrees with the pyscf SCF reference by tens of Ha for
    # localized systems such as H2O. This test catches that failure mode at the
    # AO-leaf level (sub-second), without needing the full VMC pipeline.
    # ---------------------------------------------------------------------------
    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_pgto_complex_with_negated_k_is_TR_pair(self, ao_kind: str):
        """``aos.with_negated_k()`` produces the time-reversal partner ``phi^{-k}``.

        Builds a complex Bloch pGTO leaf at ``+k`` and its ``with_negated_k()``
        sibling. Evaluates both leaves at the same set of electron positions
        and verifies ``orb_dn(r) == conj(orb_up(r))`` to ``ao_eval`` strict
        tolerance. The relation must hold for arbitrary ``r`` (not only at
        the home cell or lattice sites) because the kernel does not apply any
        spin-asymmetric phase correction itself.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre atom (offset toward a corner): under the production
        # per-atom (Omega - R_A) image truncation a CENTRED atom in a sparse
        # cell needs no lattice images (n_image == 1), so a centred fixture
        # would no longer exercise the image fold. A near-corner atom keeps the
        # interior-octant images for any localised shell (n_image > 1).
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        structure_pbc = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )

        k_plus = (0.27, -0.13, 0.41)  # generic non-Gamma, non-symmetric twist
        common = dict(
            structure_data=structure_pbc,
            nucleus_index=(0, 0, 0),
            num_ao=3,
            num_ao_prim=3,
            orbital_indices=(0, 1, 2),
            exponents=np.array([0.8, 1.6, 2.4], dtype=np.float64),
            coefficients=np.array([1.0, 0.7, 0.5], dtype=np.float64),
            angular_momentums=(0, 0, 0),
        )
        if ao_kind == "cart":
            aos_up = build_AOs_cart_pgto_complex_data(
                **common,
                polynominal_order_x=(0, 0, 0),
                polynominal_order_y=(0, 0, 0),
                polynominal_order_z=(0, 0, 0),
                k_point=k_plus,
            )
        else:
            aos_up = build_AOs_sphe_pgto_complex_data(
                **common,
                magnetic_quantum_numbers=(0, 0, 0),
                k_point=k_plus,
            )

        aos_dn = aos_up.with_negated_k()
        # k_point property must flip sign exactly; the BCS / TR-pair AO leaf
        # carries its own k_point (not via structure_data).
        assert tuple(float(c) for c in aos_dn.k_point) == tuple(-float(c) for c in aos_up.k_point)

        # Evaluate both leaves at a few non-trivial off-lattice positions and
        # check ``orb_dn(r) == conj(orb_up(r))``.
        rng = np.random.default_rng(20260620)
        r_carts = rng.uniform(-3.0, 3.0, size=(7, 3)).astype(np.float64)
        ao_up = np.asarray(compute_AOs(aos_data=aos_up, r_carts=r_carts))
        ao_dn = np.asarray(compute_AOs(aos_data=aos_dn, r_carts=r_carts))

        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(ao_dn, np.conj(ao_up), atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_pbc_file",
        ["lithium.h5", "lithium_k025.h5"],
    )
    @pytest.mark.light
    def test_build_uncontracted_aos_preserves_pgto_fields(self, trexio_pbc_file: str):
        """``_build_uncontracted_aos`` on a periodic leaf must propagate the
        pgto-only fields (``_n_image``, ``_image_truncation_tol``) and the
        Bloch ``k_point``. Rebuilding the leaf via ``type(self)(...)`` would fall
        back to dataclass defaults (``_image_truncation_tol=0.0``,
        ``_n_image=1``); the resulting h5 written by ``jqmc-tool trexio
        convert-to ... -j3 ao-*`` would then crash ``Hamiltonian_data.load_from_hdf5``
        with ``OverflowError`` inside ``_apply_image_expansion ->
        _build_image_expanded_arrays -> _primitive_image_radius`` because
        ``log_inv_tol = max(-log(0), 1.0) = +inf -> r_cut = inf ->
        int(ceil(r_cut / L_min)) + 1``.

        Counterpart end-to-end coverage:
        ``test_jqmc_tool.py::test_trexio_convert_to_pbc_j3_load_roundtrip``.
        """
        (
            _structure,
            aos_data,
            *_rest,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_pbc_file),
            store_tuple=True,
        )

        # Sanity: the loader auto-expands periodic AOs, so this is a pgto leaf.
        assert isinstance(aos_data, (AOs_cart_pgto_data, AOs_sphe_pgto_data)), (
            f"Fixture {trexio_pbc_file!r} should yield a pgto AO leaf; got {type(aos_data).__name__}."
        )

        unc = aos_data._build_uncontracted_aos()

        assert type(unc) is type(aos_data), (
            f"_build_uncontracted_aos returned {type(unc).__name__}; expected {type(aos_data).__name__}."
        )
        assert unc._image_truncation_tol == aos_data._image_truncation_tol, (
            f"_build_uncontracted_aos dropped _image_truncation_tol: "
            f"got {unc._image_truncation_tol!r}, expected {aos_data._image_truncation_tol!r}."
        )
        assert unc._n_image == aos_data._n_image, (
            f"_build_uncontracted_aos dropped _n_image: got {unc._n_image}, expected {aos_data._n_image}."
        )
        # k_point preservation: critical on the complex Bloch path (k != Gamma) so
        # the de-tiled GTO leaf saved by ``_detile_aos_data_for_save`` still
        # carries the twist, and the load-time dispatcher picks the complex
        # Bloch-pGTO leaf rather than the real one.
        assert unc.k_point == aos_data.k_point, (
            f"_build_uncontracted_aos dropped k_point: got {unc.k_point!r}, expected {aos_data.k_point!r}."
        )

        jax.clear_caches()


# ---------------------------------------------------------------------------
# Plane-wave leaf (AOs_plane_wave_data) -- structural contracts.
# ---------------------------------------------------------------------------


class TestPWLeaf:
    @pytest.mark.light
    def test_pw_compute_AOs_matches_reference(self):
        """Production ``compute_AOs`` matches the direct numpy cos/sin reference.

        Precision: the reference phase is formed in float64 and cos/sin are
        evaluated in the ``ao_eval`` dtype (see :py:func:`_pw_reference_values`).
        """
        structure = _nongto_cubic_structure()
        g_vectors, phase_is_sin = _pw_basis(structure)
        aos = AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)
        rng = np.random.default_rng(1234)
        r = rng.uniform(0.1, _NONGTO_A - 0.1, size=(7, 3)).astype(np.float64)
        got = np.asarray(compute_AOs(aos_data=aos, r_carts=r))
        ref = _pw_reference_values(g_vectors, phase_is_sin, r)
        assert got.shape == (aos.num_ao, r.shape[0])
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        atol, rtol = (
            get_tolerance("ao_eval", "strict") if is_mixed_precision_enabled() else (ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT)
        )
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)

    @pytest.mark.light
    def test_pw_num_ao_inferred(self):
        """``num_ao`` is inferred from ``g_vectors`` (1 constant + 5 cos/sin pairs)."""
        structure = _nongto_cubic_structure()
        aos = _build_pw_leaf(structure)
        assert aos.num_ao == 11
        assert aos.g_vectors.shape == (11, 3)
        assert aos.phase_is_sin.shape == (11,)

    @pytest.mark.light
    def test_pw_g_indices_roundtrip(self):
        """The builder converts Cartesian ``g_vectors`` to integer
        indices and the derived ``g_vectors`` property reconstructs them.

        Pins that the builder's ``m = round(g . cell.T / (2pi))`` conversion is
        integer-exact for Cartesian G's on the reciprocal lattice, and that the
        forward derivation ``G = m . (2pi * inv(cell).T)`` recovers the originals
        to fp64 round-off.
        """
        structure = _nongto_cubic_structure()
        g_vectors_in, _phase = _pw_basis(structure)
        aos = _build_pw_leaf(structure)

        # Storage shape: flat tuple of length 3 * num_ao.
        assert isinstance(aos._g_indices, tuple)
        assert len(aos._g_indices) == 3 * aos.num_ao
        # All entries are Python ints (no numpy types lurking).
        assert all(isinstance(x, int) for x in aos._g_indices)

        # Derived g_vectors match the builder input bit-for-bit (Cartesian G's
        # on the reciprocal lattice round-trip exactly through integer indices).
        g_recovered = np.asarray(aos.g_vectors)
        np.testing.assert_array_equal(g_recovered.shape, g_vectors_in.shape)
        np.testing.assert_allclose(g_recovered, g_vectors_in, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)

        # JAX-active form yields the same values (no `.replace(structure_data=...)`
        # yet, so this is just a forward consistency check; S2 will activate the
        # cell-AD chain through this property).
        g_jnp = np.asarray(aos._g_vectors_jnp)
        np.testing.assert_allclose(g_jnp, g_recovered, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)

    @pytest.mark.light
    def test_pw_g_indices_off_lattice_rejected(self):
        """The builder rejects Cartesian G's that are not on
        the reciprocal lattice (|m - round(m)| > 1e-9).
        """
        structure = _nongto_cubic_structure()
        # Lattice constant `_NONGTO_A` and reciprocal step 2pi/a.
        a = float(np.asarray(structure.vec_a)[0])
        # An off-lattice G with a sub-integer index component.
        bad_g = np.array([[2.5 * np.pi / a, 0.0, 0.0]], dtype=np.float64)
        bad_phase = np.array([0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="reciprocal lattice"):
            AOs_plane_wave_data.build(structure, bad_g, bad_phase)

    @pytest.mark.light
    def test_pw_value_grad_lap_consistency(self):
        """Fused ``compute_AOs_value_grad_lap`` matches the separate calls."""
        structure = _nongto_cubic_structure()
        aos = _build_pw_leaf(structure)
        rng = np.random.default_rng(1234)
        r = rng.uniform(0.1, _NONGTO_A - 0.1, size=(7, 3)).astype(np.float64)
        fused = compute_AOs_value_grad_lap(aos_data=aos, r_carts=r)
        assert all(np.all(np.isfinite(np.asarray(x))) for x in fused)
        # Documented zone contract of the fused API (issue #27): the value is returned in
        # ``ao_eval`` like ``compute_AOs``, the derivatives in ``ao_grad_lap``.
        assert fused[0].dtype == get_dtype_jnp("ao_eval"), f"val.dtype = {fused[0].dtype}"
        assert all(x.dtype == get_dtype_jnp("ao_grad_lap") for x in fused[1:]), [x.dtype for x in fused[1:]]
        val, gx, gy, gz, lap = (np.asarray(x) for x in fused)
        value_atol, value_rtol = (
            get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
            if is_mixed_precision_enabled()
            else (ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT)
        )
        np.testing.assert_allclose(val, np.asarray(compute_AOs(aos_data=aos, r_carts=r)), rtol=value_rtol, atol=value_atol)
        gx_ref, gy_ref, gz_ref = (np.asarray(g) for g in compute_AOs_grad(aos_data=aos, r_carts=r))
        np.testing.assert_allclose(gx, gx_ref, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(gy, gy_ref, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(gz, gz_ref, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            lap, np.asarray(compute_AOs_laplacian(aos_data=aos, r_carts=r)), rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE
        )

    @pytest.mark.light
    def test_pw_periodicity_under_lattice_translation(self):
        """Plane waves are invariant under integer lattice translations.

        Besides n = (1, -1, 2), the far shifts n = (101, -37, 63) and (10001, -7003, 4005)
        (up to about 7e4 bohr) check that the home-cell reduction keeps the invariance
        independent of |n| in each precision mode.
        """
        a = _NONGTO_A
        structure = _nongto_cubic_structure(a)
        aos = _build_pw_leaf(structure)
        rng = np.random.default_rng(1234)
        r = rng.uniform(0.1, a - 0.1, size=(7, 3)).astype(np.float64)
        v0 = np.asarray(compute_AOs(aos_data=aos, r_carts=r))
        for n in ((1, -1, 2), (101, -37, 63), (10001, -7003, 4005)):
            shifted = r + a * np.asarray(n, dtype=np.float64)
            v1 = np.asarray(compute_AOs(aos_data=aos, r_carts=shifted))
            np.testing.assert_allclose(v0, v1, rtol=RTOL_BIT_EXACT, atol=ATOL_FLOOR_MEDIUM, err_msg=f"n={n}")

    @pytest.mark.light
    def test_pw_far_shifted_values_match_home_cell_cos_sin_reference(self):
        """Real Gamma plane waves at a far lattice image equal cos/sin at the home-cell point.

        An unwrapped walker r + n.cell with n = (101, -37, 63), (10001, -7003, 4005) and
        (1000001, -700003, 400005) (up to about 7e6 bohr) must give exactly the plane-wave values of the home-cell point
        r, because G.n.cell is a multiple of 2 pi for every reciprocal-lattice G. The
        reference is the direct numpy cos(G.r) / sin(G.r) of :py:func:`_pw_reference_values`
        at the home-cell r (phase in float64, cos/sin in the ``ao_eval`` dtype); it never
        sees the far coordinate. The comparison uses the ``ao_eval`` strict zone policy of
        the active mode, so in mixed precision the leaf must reduce the fractional
        coordinate in fp64 before the fp32 phase.
        """
        a = _NONGTO_A
        structure = _nongto_cubic_structure(a)
        g_vectors, phase_is_sin = _pw_basis(structure)
        aos = AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)
        rng = np.random.default_rng(2626)
        r = rng.uniform(0.1, a - 0.1, size=(7, 3)).astype(np.float64)
        ref = _pw_reference_values(g_vectors, phase_is_sin, r)
        assert np.all(np.isfinite(ref)) and np.max(np.abs(ref)) > 0.5
        atol, rtol = get_tolerance("ao_eval", "strict")
        for n in ((101, -37, 63), (10001, -7003, 4005), (1000001, -700003, 400005)):
            far = r + np.asarray(n, dtype=np.float64) @ np.asarray(structure.cell, dtype=np.float64)
            got = np.asarray(compute_AOs(aos_data=aos, r_carts=far))
            assert got.shape == ref.shape and np.all(np.isfinite(got))
            np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=f"n={n}")

    @pytest.mark.light
    def test_pw_fused_value_grad_lap_returns_documented_zone_dtypes(self):
        """The fused real plane-wave API returns the documented zone dtypes.

        ``compute_AOs_value_grad_lap`` documents ``val`` in the ``ao_eval`` zone (the dtype
        ``compute_AOs`` returns) and ``gx``, ``gy``, ``gz``, ``lap`` in the ``ao_grad_lap``
        zone. The expected dtypes are read from the zone table of the active mode, not
        from the leaf. The fused value is also compared with the direct numpy cos/sin
        reference so a correct dtype cannot hide a wrong value.
        """
        structure = _nongto_cubic_structure()
        g_vectors, phase_is_sin = _pw_basis(structure)
        aos = AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)
        rng = np.random.default_rng(2727)
        r = rng.uniform(0.1, _NONGTO_A - 0.1, size=(7, 3)).astype(np.float64)
        fused = compute_AOs_value_grad_lap(aos_data=aos, r_carts=r)
        assert all(np.all(np.isfinite(np.asarray(x))) for x in fused)
        assert len(fused) == 5
        value_dtype = np.dtype(get_dtype_np("ao_eval"))
        derivative_dtype = np.dtype(get_dtype_np("ao_grad_lap"))
        assert np.dtype(fused[0].dtype) == value_dtype, f"val.dtype = {fused[0].dtype}, expected {value_dtype}"
        assert np.dtype(compute_AOs(aos_data=aos, r_carts=r).dtype) == value_dtype
        for name, x in zip(("gx", "gy", "gz", "lap"), fused[1:]):
            assert np.dtype(x.dtype) == derivative_dtype, f"{name}.dtype = {x.dtype}, expected {derivative_dtype}"
        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(np.asarray(fused[0]), _pw_reference_values(g_vectors, phase_is_sin, r), rtol=rtol, atol=atol)

    @pytest.mark.light
    def test_pw_compute_MOs_reproduces_known_function(self):
        """Agnostic MO path: ``compute_MOs`` == direct linear combination of the basis.

        Precision: the basis values are the ``ao_eval``-dtype reference of
        :py:func:`_pw_reference_values`; the linear combination is contracted
        in the ``mo_eval`` dtype, mirroring the production zone split.
        """
        structure = _nongto_cubic_structure()
        g_vectors, phase_is_sin = _pw_basis(structure)
        aos = AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)
        rng = np.random.default_rng(7)
        mo_coefficients = rng.standard_normal((3, aos.num_ao)).astype(np.float64)
        mos = MOs_data(num_mo=3, aos_data=aos, mo_coefficients=mo_coefficients)
        mos.sanity_check()
        r = rng.uniform(0.1, _NONGTO_A - 0.1, size=(7, 3)).astype(np.float64)
        got = np.asarray(compute_MOs(mos_data=mos, r_carts=r))
        mo_dtype = get_dtype_np("mo_eval")
        ref = mo_coefficients.astype(mo_dtype) @ _pw_reference_values(g_vectors, phase_is_sin, r).astype(mo_dtype)
        assert got.shape == (3, r.shape[0])
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        atol, rtol = (
            get_tolerance_min(("ao_eval", "mo_eval"), "strict")
            if is_mixed_precision_enabled()
            else (ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT, RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT)
        )
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)

    @pytest.mark.light
    def test_pw_sanity_check_rejects_bad_shapes(self):
        """Pw sanity check rejects bad shapes."""
        structure = _nongto_cubic_structure()
        # Storage is a flat tuple of length 3 * num_ao;
        # mismatched ``num_ao`` vs ``len(_g_indices) / 3`` is what
        # sanity_check now flags.
        with pytest.raises(ValueError):
            AOs_plane_wave_data(
                structure_data=structure,
                num_ao=2,
                _g_indices=(0, 0, 0, 1, 0, 0, 0, 1, 0),  # 9 entries (=3*3) vs 3*num_ao=6
                phase_is_sin=np.zeros((3,), dtype=np.float64),
            ).sanity_check()

    @pytest.mark.light
    def test_pw_sanity_check_requires_pbc(self):
        """Pw sanity check requires PBC."""
        structure_mol = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3), dtype=np.float64),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        with pytest.raises(ValueError):
            AOs_plane_wave_data(
                structure_data=structure_mol,
                num_ao=1,
                _g_indices=(0, 0, 0),
                phase_is_sin=np.zeros((1,), dtype=np.float64),
            ).sanity_check()

    @pytest.mark.light
    def test_pw_complex_flag_guard(self):
        """Pw complex flag guard."""
        structure = _nongto_cubic_structure()
        g_vectors, phase_is_sin = _pw_basis(structure)
        # Re-use the builder (which converts Cartesian g_vectors to _g_indices),
        # then mutate _complex via replace() since it is pytree_node=False.
        aos = AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)
        aos_complex = aos.replace(_complex=True)
        with pytest.raises(NotImplementedError):
            aos_complex.sanity_check()


# ---------------------------------------------------------------------------
# B-spline leaf (AOs_bspline_data) -- structural contracts.
# ---------------------------------------------------------------------------


def _bspline_interior_points(a: float = _NONGTO_A, n: int = 9) -> np.ndarray:
    rng = np.random.default_rng(2024)
    return rng.uniform(0.05, a - 0.05, size=(n, 3)).astype(np.float64)


class TestBsplineLeaf:
    @pytest.mark.light
    def test_tiled_bspline_matches_primitive_k_blocks_without_supercell_grid(self):
        """Tiled values/derivatives equal primitive Bloch blocks with linear storage."""
        primitive = _nongto_cubic_structure()
        tiling = np.diag([2, 1, 1])
        supercell = tile_structure_data(primitive, tiling)
        base = _build_bspline_leaf(primitive)
        real_coefficients = np.asarray(base.spline_coefficients, dtype=np.float64)
        # Two distinct primitive k blocks, each with two bands.  Both unfold
        # to Gamma because S @ (0,0,0) and S @ (1/2,0,0) are integral.
        blocks = (
            real_coefficients.astype(np.complex128),
            (0.7 + 0.2j) * real_coefficients + (0.1 - 0.05j) * np.roll(real_coefficients, 1, axis=1),
        )
        k_points = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))
        tiled = AOs_bspline_tiled_complex_data.build(
            structure_data=supercell,
            spline_coefficients_by_k=blocks,
            spline_shape=tuple(base.spline_shape),
            tiling_matrix=tiling,
            k_points=k_points,
            spline_order=base.spline_order,
        )
        reference_leaves = tuple(
            AOs_bspline_complex_data.build(
                structure_data=primitive,
                k_point=k_point,
                spline_coefficients=block / np.sqrt(2.0),
                spline_shape=tuple(base.spline_shape),
                spline_order=base.spline_order,
            )
            for block, k_point in zip(blocks, k_points, strict=True)
        )
        points = np.array(
            [[0.31, 0.72, 1.13], [3.43, 1.27, 4.91], [-1.17, 6.32, -2.04]],
            dtype=np.float64,
        )
        got_fields = tuple(np.asarray(field) for field in compute_AOs_value_grad_lap(tiled, points))
        reference_fields = tuple(
            np.concatenate(
                [np.asarray(compute_AOs_value_grad_lap(leaf, points)[field_index]) for leaf in reference_leaves],
                axis=0,
            )
            for field_index in range(5)
        )
        for got, reference in zip(got_fields, reference_fields, strict=True):
            np.testing.assert_allclose(
                got,
                reference,
                rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
                atol=ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
            )

        atol_debug, rtol_debug = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(
            np.asarray(compute_AOs(tiled, points)),
            tiled._compute_AOs_debug(points),
            rtol=rtol_debug,
            atol=atol_debug,
        )
        translated = points + supercell.cell[0] - 2.0 * supercell.cell[2]
        np.testing.assert_allclose(
            np.asarray(compute_AOs(tiled, translated)),
            np.asarray(compute_AOs(tiled, points)),
            rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
            atol=ATOL_FLOOR_MEDIUM,
        )
        assert tiled.spline_coefficients.shape == (4, int(np.prod(base.spline_shape)))
        assert tiled.tiling_determinant == 2
        np.testing.assert_allclose(tiled.primitive_cell_np, primitive.cell, rtol=RTOL_BIT_EXACT, atol=ATOL_BIT_EXACT)

        # Frozen tiled orbitals bypass the otherwise-dense identity MO matrix.
        mos = MOs_data.build_identity(aos_data=tiled)
        assert mos.identity
        assert np.asarray(mos.mo_coefficients).size == 0
        np.testing.assert_allclose(
            np.asarray(compute_MOs(mos, points)),
            np.asarray(compute_AOs(tiled, points)),
            rtol=RTOL_BIT_EXACT,
            atol=ATOL_BIT_EXACT,
        )
        sibling = tiled.with_negated_k()
        assert sibling.conjugate_coefficients
        assert sibling.spline_coefficients is tiled.spline_coefficients
        np.testing.assert_allclose(
            np.asarray(compute_AOs(sibling, points)),
            np.conj(np.asarray(compute_AOs(tiled, points))),
            rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
            atol=ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
        )

    @pytest.mark.light
    def test_bspline_compute_AOs_matches_numpy_evaluator(self):
        """Production ``compute_AOs`` matches an independent numpy spline evaluator.

        Precision: the reference uses the same spline table and the separable
        local-support contraction of ``evaluate_spline_orbitals``, but its
        interpolation arithmetic runs in the ``ao_eval`` complex dtype (the zone
        that evaluates the production spline values), with the fractional
        geometry reconstructed in float64 first. In full mode every zone is
        float64 and the reference runs in complex128.
        """
        structure = _nongto_cubic_structure()
        leaf = _build_bspline_leaf(structure)
        coeffs_cube = np.asarray(leaf.spline_coefficients).reshape((leaf.num_ao, *_BSPLINE_NS))
        r = _bspline_interior_points()
        got = np.asarray(compute_AOs(aos_data=leaf, r_carts=r))
        inv_cell = np.linalg.inv(structure.cell)
        frac = (r @ inv_cell) % 1.0  # (n, 3), float64 geometry
        ref = _bspline_spline_reference_values(coeffs_cube, frac, _BSPLINE_ORDER)
        assert got.shape == (leaf.num_ao, r.shape[0])
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        atol, rtol = (
            get_tolerance("ao_eval", "strict")
            if is_mixed_precision_enabled()
            else (ATOL_PYTHON_FP64_AGGREGATOR_LOOSE, RTOL_PYTHON_FP64_AGGREGATOR_LOOSE)
        )
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)

    @pytest.mark.light
    def test_bspline_reproduces_smooth_bands(self):
        """The interpolating spline reproduces the smooth bands off-knot.

        Precision: the analytic bands are evaluated in the ``ao_eval`` dtype at the
        float64-reconstructed fractional coordinates (float64 in full mode).
        """
        structure = _nongto_cubic_structure()
        leaf = _build_bspline_leaf(structure)
        r = _bspline_interior_points()
        got = np.asarray(compute_AOs(aos_data=leaf, r_carts=r))
        frac = _bspline_band_reference_frac(structure, r)
        ref = _bspline_band_functions(frac)  # (n_band, n)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        np.testing.assert_allclose(got, ref, rtol=RTOL_BIT_EXACT, atol=ATOL_INTERPOLATION_TIGHT)

    @pytest.mark.light
    def test_bspline_value_grad_lap_consistency(self):
        """Fused ``compute_AOs_value_grad_lap`` matches the separate calls."""
        structure = _nongto_cubic_structure()
        leaf = _build_bspline_leaf(structure)
        r = _bspline_interior_points()
        val, gx, gy, gz, lap = (np.asarray(x) for x in compute_AOs_value_grad_lap(aos_data=leaf, r_carts=r))
        value_atol, value_rtol = (
            get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
            if is_mixed_precision_enabled()
            else (ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT)
        )
        np.testing.assert_allclose(val, np.asarray(compute_AOs(aos_data=leaf, r_carts=r)), rtol=value_rtol, atol=value_atol)
        gx_ref, gy_ref, gz_ref = (np.asarray(g) for g in compute_AOs_grad(aos_data=leaf, r_carts=r))
        np.testing.assert_allclose(gx, gx_ref, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(gy, gy_ref, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(gz, gz_ref, rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            lap, np.asarray(compute_AOs_laplacian(aos_data=leaf, r_carts=r)), rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE
        )

    @pytest.mark.light
    def test_bspline_periodicity_under_lattice_translation(self):
        """The whole-cell periodic spline is invariant under lattice translations.

        Besides n = (1, -1, 2), the far shifts n = (101, -37, 63) and (10001, -7003, 4005)
        (up to about 7e4 bohr) check the invariance independently of |n| in each
        precision mode.
        """
        structure = _nongto_cubic_structure()
        leaf = _build_bspline_leaf(structure)
        r = _bspline_interior_points()
        v0 = np.asarray(compute_AOs(aos_data=leaf, r_carts=r))
        for n in ((1, -1, 2), (101, -37, 63), (10001, -7003, 4005)):
            shifted = r + _NONGTO_A * np.asarray(n, dtype=np.float64)
            v1 = np.asarray(compute_AOs(aos_data=leaf, r_carts=shifted))
            np.testing.assert_allclose(v0, v1, rtol=RTOL_BIT_EXACT, atol=ATOL_FLOOR_MEDIUM, err_msg=f"n={n}")

    @pytest.mark.light
    def test_bspline_quintic_contraction_pairs_each_axis_with_its_weights(self):
        """The quintic control-point contraction pairs axis x, y, z with wx, wy, wz.

        Two levels, both in the ``ao_eval`` dtype of the active mode (the CPU fp32
        quintic path in mixed precision, the einsum path in full precision):

        1. ``AOs_bspline_data._contract_control_points`` on seeded control points
           ``cg[b, p, q, r]`` and three different weight vectors is compared with the
           float64 numpy tensor contraction ``sum_pqr cg[b,p,q,r] wx[p] wy[q] wz[r]``.
        2. ``compute_AOs`` of a quintic leaf on an anisotropic (8, 10, 12) grid with two
           bands that are not symmetric under any axis exchange is compared with the
           independent separable evaluator :py:func:`_bspline_spline_reference_values`
           (one Cartesian axis at a time, its own weights and its own grid size).

        A contraction that pairs an axis with the weights of another axis changes both
        results by O(0.1). Each level is judged with the ``ao_eval`` strict zone policy.
        """
        dtype = get_dtype_np("ao_eval")
        atol, rtol = get_tolerance("ao_eval", "strict")
        rng = np.random.default_rng(2525)
        cg = rng.standard_normal((3, 6, 6, 6)).astype(dtype)
        wx, wy, wz = (_bspline_weights_zone(np.asarray([f], dtype=dtype), 5)[0] for f in (0.11, 0.47, 0.83))
        assert min(np.max(np.abs(wx - wy)), np.max(np.abs(wy - wz)), np.max(np.abs(wx - wz))) > 1.0e-2
        got = np.asarray(
            AOs_bspline_data._contract_control_points(jnp.asarray(cg), jnp.asarray(wx), jnp.asarray(wy), jnp.asarray(wz))
        )
        ref = np.einsum(
            "bpqr,p,q,r->b",
            cg.astype(np.float64),
            wx.astype(np.float64),
            wy.astype(np.float64),
            wz.astype(np.float64),
        )
        assert np.max(np.abs(ref)) > 1.0e-1
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)

        structure = _nongto_cubic_structure()
        shape = (8, 10, 12)
        fx, fy, fz = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
        two_pi = 2.0 * np.pi
        bands = np.stack(
            (
                np.sin(two_pi * fx) + 0.5 * np.cos(two_pi * fy) + 0.25 * np.sin(two_pi * fz),
                np.cos(two_pi * (fx + 2.0 * fy)) + 0.3 * np.sin(two_pi * fz),
            )
        )
        coeffs_cube = bspline_prefilter_periodic(bands.astype(np.complex128), 5)
        leaf = AOs_bspline_data.build(structure, coeffs_cube.reshape(coeffs_cube.shape[0], -1), shape, 5)
        r = _bspline_interior_points()
        frac = (r @ np.linalg.inv(np.asarray(structure.cell, dtype=np.float64))) % 1.0
        ref_values = _bspline_spline_reference_values(coeffs_cube, frac, 5)
        got_values = np.asarray(compute_AOs(aos_data=leaf, r_carts=r))
        assert got_values.shape == ref_values.shape and np.max(np.abs(ref_values)) > 0.5
        np.testing.assert_allclose(got_values, ref_values, rtol=rtol, atol=atol)

    @pytest.mark.light
    def test_bspline_compute_MOs_identity_returns_bands(self):
        """Agnostic MO path: identity coefficients reproduce the band values."""
        structure = _nongto_cubic_structure()
        leaf = _build_bspline_leaf(structure)
        n_band = leaf.num_ao
        mo_coefficients = np.eye(n_band, dtype=np.float64)
        mos = MOs_data(num_mo=n_band, aos_data=leaf, mo_coefficients=mo_coefficients)
        mos.sanity_check()
        r = _bspline_interior_points()
        got = np.asarray(compute_MOs(mos_data=mos, r_carts=r))
        ref = np.asarray(compute_AOs(aos_data=leaf, r_carts=r))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT, atol=ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT
        )

    @pytest.mark.light
    def test_bspline_cubic_order_also_works(self):
        """The cubic (order 3) spline also reproduces the smooth bands.

        Precision: the analytic bands are evaluated in the ``ao_eval`` dtype at the
        float64-reconstructed fractional coordinates (float64 in full mode).
        """
        structure = _nongto_cubic_structure()
        leaf = _build_bspline_leaf(structure, order=3)
        r = _bspline_interior_points()
        got = np.asarray(compute_AOs(aos_data=leaf, r_carts=r))
        frac_pts = _bspline_band_reference_frac(structure, r)
        ref = _bspline_band_functions(frac_pts)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        np.testing.assert_allclose(got, ref, rtol=RTOL_BIT_EXACT, atol=ATOL_INTERPOLATION_RELAXED)

    @pytest.mark.light
    def test_bspline_sanity_check_rejects_bad_shapes(self):
        """Bspline sanity check rejects bad shapes."""
        structure = _nongto_cubic_structure()
        with pytest.raises(ValueError):
            AOs_bspline_data(
                structure_data=structure,
                num_ao=2,
                spline_coefficients=np.zeros((2, 10), dtype=np.float64),  # 10 != prod(shape)
                spline_shape=(12, 12, 12),
                spline_order=5,
            ).sanity_check()

    @pytest.mark.light
    def test_bspline_sanity_check_requires_pbc(self):
        """Bspline sanity check requires PBC."""
        structure_mol = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3), dtype=np.float64),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        n_pts = int(np.prod(_BSPLINE_NS))
        with pytest.raises(ValueError):
            AOs_bspline_data(
                structure_data=structure_mol,
                num_ao=1,
                spline_coefficients=np.zeros((1, n_pts), dtype=np.float64),
                spline_shape=_BSPLINE_NS,
                spline_order=5,
            ).sanity_check()

    @pytest.mark.light
    def test_bspline_complex_flag_guard(self):
        """Bspline complex flag guard."""
        structure = _nongto_cubic_structure()
        n_pts = int(np.prod(_BSPLINE_NS))
        with pytest.raises(NotImplementedError):
            AOs_bspline_data(
                structure_data=structure,
                num_ao=1,
                spline_coefficients=np.zeros((1, n_pts), dtype=np.float64),
                spline_shape=_BSPLINE_NS,
                spline_order=5,
                _complex=True,
            ).sanity_check()


# ---------------------------------------------------------------------------
# Periodic AO image-sum
# guards. The image sum's job is to keep the periodised AO CONTINUOUS across
# the cell boundary (psi(r)=psi(r+T) is trivially true because compute_AOs
# wraps electrons into the home cell, so it does NOT exercise the image set).
# A truncation that drops a NEEDED lattice image breaks this seam continuity;
# these tests pin it so the per-atom truncation cannot regress it.
# ---------------------------------------------------------------------------


class TestPGTOSeamContinuity:
    """Direct physical guard for the periodic AO image sum.

    The thing that crosses the cell boundary during MCMC is the ELECTRON, so
    the harsh point is an electron sitting AT the seam (a face / edge / corner)
    -- there it is served by the atom's image in the NEIGHBOUR cell, not by the
    home copy. The atom sits at a generic INTERIOR position offset toward the
    (---) corner (NOT on a boundary -- that would be a degenerate, partly
    symmetric placement), so each + face / edge / corner is reached purely
    through a lattice image and dropping that image is catastrophic.

    Measured discrimination on the current 27-image fixture: a complete image
    sum holds the seam to rel ~1e-4, whereas dropping the neighbour images
    jumps the seam to rel ~1.0 (one side of the seam falls to ~0 while the
    other stays O(1)). ``SEAM_RTOL = 1e-2`` cleanly separates "continuous"
    from "a needed image is missing".
    """

    EPS = 1.0e-5
    SEAM_RTOL = RTOL_PBC_SEAM_CONTINUITY  # complete sum ~1e-4; a dropped needed image -> ~1.0
    L = 10.0
    # Generic interior atom, offset toward the (---) corner and asymmetric to
    # avoid any accidental mirror symmetry across a seam.
    ATOM_FRAC = (0.10, 0.12, 0.08)

    def _seam_pgto(self, ao_kind: str):
        """s+p+d pGTO leaf on a single interior atom (see ``ATOM_FRAC``)."""
        cell = CELL_FIXTURES["cubic_10bohr"]
        positions = (np.asarray(self.ATOM_FRAC, dtype=np.float64) * self.L).reshape(1, 3)
        structure = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        aos = _make_aos_data(structure, ao_kind, l_max=2, uniform=True)
        expanded = _apply_image_expansion(aos, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
        assert expanded._n_image > 1, "fixture must exercise a non-trivial image sum"
        return expanded, positions[0]

    def _assert_seam_continuous(self, leaf, r_hi, r_lo, ao_kind):
        v_hi = np.asarray(compute_AOs(aos_data=leaf, r_carts=np.array([r_hi], dtype=np.float64)))
        v_lo = np.asarray(compute_AOs(aos_data=leaf, r_carts=np.array([r_lo], dtype=np.float64)))
        scale = float(np.max(np.abs(np.concatenate([v_hi, v_lo]))))
        assert scale > 1.0e-3, f"{ao_kind}: seam fixture must place O(1) AO density at the seam (got {scale:.2e})"
        rel = float(np.max(np.abs(v_hi - v_lo))) / scale
        assert rel < self.SEAM_RTOL, (
            f"{ao_kind}: AO discontinuous across the cell seam {r_hi} -> {r_lo} "
            f"(rel={rel:.2e} >= {self.SEAM_RTOL}); a needed lattice image is missing"
        )

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_pgto_continuity_across_each_face(self, ao_kind: str):
        """The ELECTRON at each cell face is served by the atom's image and
        evaluates continuously with its periodic copy at the opposite face.

        The transverse coordinates track the atom so the seam point carries
        O(1) AO density (an electron far from any atom-image would be a
        trivially-zero, insensitive crossing).
        """
        leaf, (ax, ay, az) = self._seam_pgto(ao_kind)
        L, eps = self.L, self.EPS
        crossings = [
            ([L - eps, ay, az], [eps, ay, az]),  # +x / -x seam
            ([ax, L - eps, az], [ax, eps, az]),  # +y / -y seam
            ([ax, ay, L - eps], [ax, ay, eps]),  # +z / -z seam
        ]
        for r_hi, r_lo in crossings:
            self._assert_seam_continuous(leaf, r_hi, r_lo, ao_kind)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_pgto_continuity_across_corner_and_edges(self, ao_kind: str):
        """Worst case: the ELECTRON at the triple-corner and on the edges
        (two or three seams crossed at once -- where an over-aggressive
        truncation is most likely to drop a needed image)."""
        leaf, (ax, ay, az) = self._seam_pgto(ao_kind)
        L, eps = self.L, self.EPS
        pairs = [
            ([L - eps, L - eps, L - eps], [eps, eps, eps]),  # corner (3 seams)
            ([L - eps, L - eps, az], [eps, eps, az]),  # xy edge
            ([L - eps, ay, L - eps], [eps, ay, eps]),  # xz edge
            ([ax, L - eps, L - eps], [ax, eps, eps]),  # yz edge
        ]
        for r_hi, r_lo in pairs:
            self._assert_seam_continuous(leaf, r_hi, r_lo, ao_kind)
        jax.clear_caches()


class TestPerAtomRaggedLayout:
    """The host-side ragged per-atom image
    layout. Each atom is tiled by only its own image set; atoms are bucketed by
    multiplicity so the device fold is a few dense ``reshape(m, K, -1).sum(0)``
    ops. These tests pin the LAYOUT:
    the per-atom image counts, the kernel-size saving vs the (A) uniform union,
    and -- the crux -- that the bucketed dense fold reproduces the direct
    per-home-AO image sum byte-for-byte.
    """

    L = 10.0

    def _multi_bucket_leaf(self):
        """A periodic cell with a centred / face / corner atom (one s+p shell
        each) so the localised per-atom sets differ (m = 1 / 2 / 8) -> 3 buckets."""
        cell = CELL_FIXTURES["cubic_10bohr"]
        positions = np.array([[5.0, 5.0, 5.0], [0.0, 5.0, 5.0], [0.0, 0.0, 0.0]], dtype=np.float64)  # centre, face, corner
        structure = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1, 1, 1),
            element_symbols=("H", "H", "H"),
            atomic_labels=("H1", "H2", "H3"),
        )
        # One s + one p (3 m-components) shell per atom, localised alpha so the
        # per-atom floor (centre 1 / face 2 / corner 8) bites.
        nao_per_atom = 4
        num_ao = 3 * nao_per_atom
        ang = (0, 1, 1, 1) * 3
        mag = (0, -1, 0, 1) * 3
        nucleus = tuple(a for a in range(3) for _ in range(nao_per_atom))
        leaf = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure,
            nucleus_index=nucleus,
            num_ao=num_ao,
            num_ao_prim=num_ao,
            orbital_indices=tuple(range(num_ao)),
            exponents=np.full(num_ao, 100.0),
            coefficients=np.ones(num_ao),
            angular_momentums=ang,
            magnetic_quantum_numbers=mag,
        )
        return leaf, structure

    @pytest.mark.light
    def test_per_atom_counts_and_kernel_saving(self):
        """Per atom counts and kernel saving."""
        leaf, structure = self._multi_bucket_leaf()
        lay = _build_ragged_image_layout(
            leaf, structure, tol=1.0e-12
        )  # tol-ok: algorithm input, result asserted against named/pinned values
        # Localised alpha=100, one alpha per atom so per-shell == per-atom here:
        # centre -> 1, face -> 2, corner -> 8 images per AO.
        assert sorted({int(x) for x in lay["ao_n_image"]}) == [1, 2, 8]
        # Ragged kernel is strictly smaller than the uniform union tiling.
        uniform = lay["n_image_union"] * lay["num_ao"]
        assert lay["num_kernel"] < uniform
        # M = sum_A m_A * k_A = (1 + 2 + 8) * 4 = 44.
        assert lay["num_kernel"] == (1 + 2 + 8) * 4

    @pytest.mark.light
    def test_bucketed_fold_matches_direct_sum(self):
        """The dense per-bucket reshape-sum reproduces the segment-sum-free
        per-home-AO image fold exactly (the load-bearing (B) invariant)."""
        leaf, structure = self._multi_bucket_leaf()
        lay = _build_ragged_image_layout(
            leaf, structure, tol=1.0e-12
        )  # tol-ok: algorithm input, result asserted against named/pinned values
        M, num_ao = lay["num_kernel"], lay["num_ao"]
        n_cols = 5
        rng = np.random.default_rng(20260627)
        kernel = rng.standard_normal((M, n_cols)).astype(np.float64)

        # Direct (reference) fold: out[h] = sum over kernel-AOs mapping to h.
        kth = lay["kernel_to_home"]
        direct = np.zeros((num_ao, n_cols), dtype=np.float64)
        np.add.at(direct, kth, kernel)

        # Bucketed dense fold: per bucket reshape(m, K, -1).sum(0), scattered to
        # its home AOs (exactly what the device kernel will do -- no segment_sum).
        bucketed = np.zeros((num_ao, n_cols), dtype=np.float64)
        for b in lay["buckets"]:
            m, K, off = b["m"], b["K"], b["kernel_offset"]
            block = kernel[off : off + m * K].reshape(m, K, n_cols).sum(axis=0)
            bucketed[b["home_aos"]] = block
        np.testing.assert_array_equal(bucketed, direct)

    @pytest.mark.light
    def test_kernel_images_are_each_atoms_own_set(self):
        """Every kernel-AO's image triple belongs to ITS atom's per-atom set,
        and each home AO appears exactly its atom's m_A times."""
        leaf, structure = self._multi_bucket_leaf()
        lay = _build_ragged_image_layout(
            leaf, structure, tol=1.0e-12
        )  # tol-ok: algorithm input, result asserted against named/pinned values
        from jqmc.structure import _primitive_image_list_indices_per_atom as pa

        cell = np.asarray(structure.cell, dtype=np.float64)
        inv = np.linalg.inv(cell)
        frac = np.mod(np.asarray(structure._positions_cart_np, dtype=np.float64) @ inv, 1.0)
        nuc = np.asarray(leaf.nucleus_index, dtype=np.int64)
        kth, kimg = lay["kernel_to_home"], lay["kernel_image_index"]
        # count of kernel-AOs per home AO == that AO's per-shell multiplicity
        counts = np.bincount(kth, minlength=lay["num_ao"])
        np.testing.assert_array_equal(counts, lay["ao_n_image"])
        # every kernel image is in its atom's per-atom set (alpha=100, l<=1)
        for k in range(lay["num_kernel"]):
            atom = int(nuc[int(kth[k])])
            sset = {tuple(int(c) for c in r) for r in pa(cell, 100.0, 1, frac[atom], 1.0e-12)}
            assert tuple(int(c) for c in kimg[k]) in sset

    @pytest.mark.light
    def test_ragged_fold_matches_direct_sum_jax(self):
        """(B) step 2: the JAX device fold reproduces the direct per-home-AO
        image sum (the JAX twin of test_bucketed_fold_matches_direct_sum) and
        is jittable (static bucket geometry -> no segment_sum)."""
        leaf, structure = self._multi_bucket_leaf()
        lay = _build_ragged_image_layout(
            leaf, structure, tol=1.0e-12
        )  # tol-ok: algorithm input, result asserted against named/pinned values
        M, num_ao = lay["num_kernel"], lay["num_ao"]
        rng = np.random.default_rng(424242)
        kernel = rng.standard_normal((M, 7)).astype(np.float64)

        direct = np.zeros((num_ao, 7), dtype=np.float64)
        np.add.at(direct, lay["kernel_to_home"], kernel)

        buckets, fold_perm = lay["buckets"], lay["fold_perm"]
        out = np.asarray(_fold_image_axis_ragged(jnp.asarray(kernel), buckets, fold_perm))
        np.testing.assert_allclose(out, direct, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)

        # jittable: bucket geometry closes over as static; fold_perm constant-folds.
        folded = jax.jit(lambda k: _fold_image_axis_ragged(k, buckets, fold_perm))(jnp.asarray(kernel))
        np.testing.assert_allclose(np.asarray(folded), direct, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)

    @pytest.mark.light
    def test_ragged_fold_reduces_to_uniform_single_bucket(self):
        """Bridge to (A): a single atom is one bucket, so the ragged fold must
        reproduce the uniform ``reshape(m, num_ao, -1).sum(0)`` of
        ``_fold_image_axis`` for that ordering."""
        cell = CELL_FIXTURES["cubic_10bohr"]
        structure = make_pbc_structure(
            cell,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),  # corner -> m = 8
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        leaf = _make_aos_data(structure, "sphe", l_max=2, uniform=True)
        lay = _build_ragged_image_layout(
            leaf, structure, tol=1.0e-12
        )  # tol-ok: algorithm input, result asserted against named/pinned values
        assert len(lay["buckets"]) == 1
        m = lay["buckets"][0]["m"]
        num_ao = lay["num_ao"]
        rng = np.random.default_rng(99)
        kernel = rng.standard_normal((m * num_ao, 4)).astype(np.float64)

        ragged = np.asarray(_fold_image_axis_ragged(jnp.asarray(kernel), lay["buckets"], lay["fold_perm"]))
        uniform = np.asarray(jnp.asarray(kernel).reshape(m, num_ao, -1).sum(axis=0))
        np.testing.assert_allclose(ragged, uniform, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_ragged_compute_matches_uniform_A(self, ao_kind: str):
        """(B) step 3 (compute equivalence): evaluating each kernel-AO at its
        own image centre (a molecular leaf at the M ragged centres) and folding
        with the bucketed ragged fold reproduces the production (A) periodic AO
        ``compute_AOs`` to the truncation envelope. Proves the ragged layout +
        per-kernel centres + fold are compute-correct BEFORE the pgto-leaf
        surgery -- the molecular leaf stands in as the kernel evaluator.
        """
        from jqmc.structure import _wrap_into_cell_if_pbc

        cell = CELL_FIXTURES["cubic_10bohr"]
        # Off-centre multi-atom periodic fixture (mixed per-atom multiplicities).
        positions = np.array([[1.0, 1.2, 0.8], [4.7, 5.1, 5.3], [0.3, 9.6, 0.4]], dtype=np.float64)
        structure = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1, 1, 1),
            element_symbols=("H", "H", "H"),
            atomic_labels=("H1", "H2", "H3"),
        )
        nao_per_atom = 4  # s + p (uniform K=1)
        num_ao = 3 * nao_per_atom
        ang = (0, 1, 1, 1) * 3
        mag = (0, -1, 0, 1) * 3
        nucleus = tuple(a for a in range(3) for _ in range(nao_per_atom))
        cart_po = {  # cart polynomial orders for s,px,py,pz
            "x": (0, 1, 0, 0) * 3,
            "y": (0, 0, 1, 0) * 3,
            "z": (0, 0, 0, 1) * 3,
        }
        common = dict(
            structure_data=structure,
            nucleus_index=nucleus,
            num_ao=num_ao,
            num_ao_prim=num_ao,
            orbital_indices=tuple(range(num_ao)),
            exponents=np.full(num_ao, 1.0),
            coefficients=np.ones(num_ao),
            angular_momentums=ang,
        )
        if ao_kind == "sphe":
            base = AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=mag)
        else:
            base = AOs_cart_gto_data._from_full_basis(
                **common, polynominal_order_x=cart_po["x"], polynominal_order_y=cart_po["y"], polynominal_order_z=cart_po["z"]
            )

        # (A) reference periodic leaf + values.
        leaf_A = _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
        rng = np.random.default_rng(7)
        r = rng.uniform(0.0, 10.0, size=(9, 3)).astype(np.float64)
        ref = np.asarray(compute_AOs(aos_data=leaf_A, r_carts=r))  # (num_ao, N_e)

        # (B) ragged: build the layout (same tol as (A)) and a MOLECULAR leaf
        # whose M AOs sit at R_atom + T_image -- the kernel evaluator stand-in.
        lay = _build_ragged_image_layout(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
        M = lay["num_kernel"]
        kth = np.asarray(lay["kernel_to_home"])
        kimg = np.asarray(lay["kernel_image_index"], dtype=np.float64)
        R_mol = np.asarray(structure._positions_cart_np, dtype=np.float64)
        cell_np = np.asarray(structure.cell, dtype=np.float64)
        base_nuc = np.asarray(base.nucleus_index)
        centres = R_mol[base_nuc[kth]] + kimg @ cell_np  # (M, 3)
        k_struct = Structure_data(
            positions=centres,
            pbc_flag=False,
            atomic_numbers=tuple([1] * M),
            element_symbols=tuple(["H"] * M),
            atomic_labels=tuple([f"H{i}" for i in range(M)]),
        )
        k_common = dict(
            structure_data=k_struct,
            nucleus_index=tuple(range(M)),
            num_ao=M,
            num_ao_prim=M,
            orbital_indices=tuple(range(M)),
            exponents=np.full(M, 1.0),
            coefficients=np.ones(M),
            angular_momentums=tuple(int(ang[h]) for h in kth),
        )
        if ao_kind == "sphe":
            kleaf = AOs_sphe_gto_data._from_full_basis(**k_common, magnetic_quantum_numbers=tuple(int(mag[h]) for h in kth))
        else:
            kleaf = AOs_cart_gto_data._from_full_basis(
                **k_common,
                polynominal_order_x=tuple(int(cart_po["x"][h]) for h in kth),
                polynominal_order_y=tuple(int(cart_po["y"][h]) for h in kth),
                polynominal_order_z=tuple(int(cart_po["z"][h]) for h in kth),
            )
        # (A) wraps r into the home cell; replicate before the (non-wrapping)
        # molecular kernel, then fold with the ragged bucketed fold.
        r_wrapped = np.asarray(_wrap_into_cell_if_pbc(structure, jnp.asarray(r)))
        kernel_vals = compute_AOs(aos_data=kleaf, r_carts=r_wrapped)  # (M, N_e)
        got = np.asarray(_fold_image_axis_ragged(kernel_vals, lay["buckets"], lay["fold_perm"]))

        assert got.shape == ref.shape == (num_ao, r.shape[0])
        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(
            got, ref, atol=max(atol, ATOL_AO_IMAGE_TRUNCATION_SUM_FACTOR * AO_IMAGE_TRUNCATION_ATOL), rtol=rtol
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
    def test_ragged_h5_detile_round_trips(self, ao_kind: str):
        """(B) step 2 of 5 -- h5. De-tiling a ragged leaf recovers the molecular
        base (so a checkpoint of a ragged run round-trips), and re-expanding the
        de-tiled base reproduces the ragged leaf's AO values."""
        from jqmc.hamiltonians import _detile_aos_data_for_save

        cell = CELL_FIXTURES["cubic_10bohr"]
        positions = np.array([[1.0, 1.2, 0.8], [4.7, 5.1, 5.3], [0.3, 9.6, 0.4]], dtype=np.float64)
        structure = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1, 1, 1),
            element_symbols=("H", "H", "H"),
            atomic_labels=("H1", "H2", "H3"),
        )
        nao_per_atom = 4
        num_ao = 3 * nao_per_atom
        ang = (0, 1, 1, 1) * 3
        mag = (0, -1, 0, 1) * 3
        po = {"x": (0, 1, 0, 0) * 3, "y": (0, 0, 1, 0) * 3, "z": (0, 0, 0, 1) * 3}
        nucleus = tuple(a for a in range(3) for _ in range(nao_per_atom))
        common = dict(
            structure_data=structure,
            nucleus_index=nucleus,
            num_ao=num_ao,
            num_ao_prim=num_ao,
            orbital_indices=tuple(range(num_ao)),
            exponents=np.full(num_ao, 1.0),
            coefficients=np.ones(num_ao),
            angular_momentums=ang,
        )
        if ao_kind == "sphe":
            base = AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=mag)
        else:
            base = AOs_cart_gto_data._from_full_basis(
                **common, polynominal_order_x=po["x"], polynominal_order_y=po["y"], polynominal_order_z=po["z"]
            )
        leaf_B = _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
        detiled = _detile_aos_data_for_save(leaf_B)
        detiled.sanity_check()
        assert not getattr(detiled, "_is_ragged", False)
        assert detiled.num_ao == base.num_ao

        rng = np.random.default_rng(321)
        r = rng.uniform(0.0, 10.0, size=(6, 3)).astype(np.float64)
        atol, rtol = get_tolerance("ao_eval", "strict")
        # de-tiled molecular base reproduces the original molecular base values
        np.testing.assert_allclose(
            np.asarray(compute_AOs(aos_data=detiled, r_carts=r)),
            np.asarray(compute_AOs(aos_data=base, r_carts=r)),
            atol=atol,
            rtol=rtol,
        )
        # re-expansion of the de-tiled base reproduces the ragged leaf
        leaf_B2 = _apply_image_expansion(detiled, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
        np.testing.assert_allclose(
            np.asarray(compute_AOs(aos_data=leaf_B2, r_carts=r)),
            np.asarray(compute_AOs(aos_data=leaf_B, r_carts=r)),
            atol=atol,
            rtol=rtol,
        )
        jax.clear_caches()


class TestPGTODistinctCentreTable:
    """The angular-table centre list holds each DISTINCT centre once.

    A periodic pGTO leaf's ``nucleus_index`` is the identity over the M KERNEL
    AOs, so the inherited "unique centre" table had one row per kernel AO:
    every angular component and every shell sharing an atom+image repeated the
    same ``R + T``. Measured 5316 -> 556 rows on silicon.

    Two traps this pins, both of which broke a first attempt:
      * the flat ``(centre, l_m)`` gather exists in TWO index conventions --
        AO-level (value-only kernel) and primitive-level (value+grad+lap).
        Remapping one and not the other strides the deduplicated table with
        un-deduplicated rows, and only the sphe <-> cart conversion notices.
      * the dedup key must be the STATIC ``(atom, image)`` pair; keying on the
        centre coordinates raises ``TracerArrayConversionError`` under jit
        because they are a traced pytree leaf.
    """

    @pytest.mark.light
    @pytest.mark.parametrize(
        "trexio_file, periodic",
        [("silicon.h5", True), ("lithium_k025.h5", True), ("water_ccecp_ccpvqz.h5", False)],
    )
    def test_distinct_centre_table_preserves_every_centre(self, trexio_file, periodic):
        """Distinct centre table preserves every centre."""
        (_structure, aos_data, mos_up, *_rest) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        # Cover the CONVERTED Cartesian leaf too, not just the stored one.
        for leaf in (aos_data, mos_up.to_cartesian().aos_data):
            inherited = np.asarray(leaf._atomic_center_carts_unique_jnp)
            distinct = np.asarray(leaf._uq_center_carts_jnp)

            # Both index conventions must land on the same centre as before.
            np.testing.assert_array_equal(
                distinct[np.asarray(leaf._uq_center_index_prim_jnp)],
                inherited[np.asarray(leaf._nucleus_index_prim_jnp)],
            )
            np.testing.assert_array_equal(
                distinct[np.asarray(leaf._uq_center_index_ao_jnp)],
                inherited[np.asarray(leaf._nucleus_index_jnp)],
            )
            assert np.unique(distinct, axis=0).shape[0] == distinct.shape[0]
            if periodic:
                assert distinct.shape[0] < inherited.shape[0], "dedup is inert on a periodic leaf"
            else:
                assert distinct.shape[0] == inherited.shape[0]


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)


@pytest.mark.light
def test_composite_complex_gram_is_conjugated_not_bilinear():
    """The mixed complex pGTO+PW Gram must be ``int conj(chi_mu) chi_nu``.

    Every registered overlap fixture was either a pure plane-wave composite
    (short-circuited by the analytic branch before the quadrature) or a Gamma
    pGTO leaf (real values, where conjugation is a no-op), so dropping the
    conjugation in the quadrature -- ``vals.conj() @ vals.T`` -> ``vals @
    vals.T`` -- changed nothing any test could see. The bilinear form is
    symmetric rather than Hermitian, and the final ``0.5 * (S + S^H)``
    hermitization then HIDES the defect by construction, leaving a wrong
    S-metric behind orthogonalization, determinants and energies (AUD2-008).

    The oracles here are properties of a Gram matrix that the bilinear form
    does not have: a real positive diagonal equal to ``int |chi|^2`` and a
    positive-definite spectrum.

    Production assembles the matrix from exact blocks; the
    reference is an independent 96^3 cell quadrature of the composite's own
    AO values (each mode's own AO precision, contracted in complex128) with an
    explicit conjugation of the bra, judged by the strict ao_eval/det_eval
    policy.
    """
    import jax.numpy as jnp

    from jqmc.atomic_orbital import (
        AOs_composite_data,
        AOs_plane_wave_complex_data,
        _compute_overlap_matrix_composite,
        compute_AOs,
    )
    from jqmc.trexio_wrapper import read_trexio_file

    stem = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_pbc_ae_k025_cart.h5")
    (sd, aos_pgto, _mu, _md, _gem, _cpd) = read_trexio_file(stem, store_tuple=True)
    assert sd.pbc_flag and aos_pgto.k_point is not None, "fixture must be a genuine non-Gamma periodic pGTO leaf"
    pw = AOs_plane_wave_complex_data(
        structure_data=sd, k_point=aos_pgto.k_point, num_ao=3, _g_indices=(0, 0, 0, 1, 0, 0, 0, 1, 0)
    )
    composite = AOs_composite_data(children=(aos_pgto, pw))
    n_pgto = int(aos_pgto.num_ao)

    S = np.asarray(_compute_overlap_matrix_composite(composite))

    # Rebuild the overlap independently as a fine cell quadrature of the AO
    # values with an explicit conjugation of the bra (the definition, written
    # out). Production assembles exact blocks (analytic lattice
    # pGTO overlap, V for the plane waves, closed-form Gaussian Fourier
    # transform across), so the quadrature must resolve the H2 Gaussians
    # (exponents <= 13; the pGTO x pGTO product needs the finest grid).
    grid = (96, 96, 96)
    cell = np.asarray(sd.cell, dtype=np.float64)
    axes = [(np.arange(n, dtype=np.float64) + 0.5) / n for n in grid]
    fa, fb, fc = np.meshgrid(*axes, indexing="ij")
    r_frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1)
    vals = np.asarray(compute_AOs(composite, jnp.asarray(r_frac @ cell, dtype=jnp.float64))).astype(np.complex128)
    volume_element = float(np.abs(np.linalg.det(cell))) / float(np.prod(grid))
    S_ref = volume_element * (vals.conj() @ vals.T)

    # Non-vacuity: the fixture must actually carry complex AO values, otherwise
    # conjugation is a no-op and this whole test proves nothing.
    assert float(np.max(np.abs(vals.imag))) > 1.0e-3, "AO values are real: the conjugation is untested"
    cross = S[:n_pgto, n_pgto:]
    assert float(np.max(np.abs(cross.imag))) > 1.0e-6, "pGTO x PW cross block has no imaginary content"

    # The pGTO rows (pGTO block and pGTO x PW cross block, where the
    # conjugation acts). The PW x PW block is the exact V delta_{G,G'}; its
    # off-diagonal zeros come from phase cancellations that fp32 AO samples
    # (mixed mode) reproduce only to the sample rounding, so it is pinned by
    # its diagonal below and by the analytic plane-wave tests.
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    np.testing.assert_allclose(S[:n_pgto], S_ref[:n_pgto], atol=atol, rtol=rtol)

    # Gram properties the unconjugated bilinear form does not satisfy.
    diag_norm = volume_element * np.einsum("ig,ig->i", vals.conj(), vals).real
    np.testing.assert_allclose(np.diag(S).real, diag_norm, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.diag(S).imag, np.zeros(S.shape[0]), atol=atol, rtol=rtol)
    np.testing.assert_allclose(S, S.conj().T, atol=atol, rtol=rtol)
    assert float(np.min(np.linalg.eigvalsh(S))) > 0.0, "the overlap matrix must be positive definite"


def _cartesian_trexio_scale_reference(leaf, points):
    """Evaluate the differentiated Cartesian primitive formula in NumPy."""
    value = np.zeros((leaf.num_ao, len(points)))
    gradient = np.zeros((3, leaf.num_ao, len(points)))
    laplacian = np.zeros_like(value)
    powers = tuple(zip(leaf.polynominal_order_x, leaf.polynominal_order_y, leaf.polynominal_order_z, strict=True))
    centers = np.asarray(leaf.structure_data.positions)[np.asarray(leaf.nucleus_index)]
    for ao, radial in zip(leaf.orbital_indices, leaf.ao_prim_to_shell_prim, strict=True):
        a = float(leaf.radial.exponents[radial])
        c = float(leaf.radial.coefficients[radial]) * leaf.ao_coefficient_scales[ao]
        n = powers[ao]
        ratio = math.prod(math.factorial(power) / math.factorial(2 * power) for power in n)
        normalization = math.sqrt((2 * a / math.pi) ** 1.5 * (8 * a) ** sum(n) * ratio)
        xyz = (points - centers[ao]).T
        polynomial = [x**power for x, power in zip(xyz, n, strict=True)]
        envelope = c * normalization * np.exp(-a * np.sum(xyz**2, axis=0))
        value[ao] += envelope * polynomial[0] * polynomial[1] * polynomial[2]
        for axis, (x, power, poly) in enumerate(zip(xyz, n, polynomial, strict=True)):
            other = polynomial[(axis + 1) % 3] * polynomial[(axis + 2) % 3]
            first = power * x ** (power - 1) if power else 0.0
            second = power * (power - 1) * x ** (power - 2) if power >= 2 else 0.0
            gradient[axis, ao] += envelope * other * (first - 2 * a * x * poly)
            laplacian[ao] += envelope * other * (second - 4 * a * x * first + (4 * a**2 * x**2 - 2 * a) * poly)
    return value, gradient, laplacian


@pytest.mark.light
@pytest.mark.parametrize("fixture", ["N_ae_ccpvdz_cart.h5", "N2_ecp_ccpvtz_cart.h5"])
def test_cartesian_trexio_component_scales_match_closed_form(fixture):
    """Check standalone and fused AO evaluation against scaled Gaussian primitives."""
    _, leaf, _, _, _, _ = read_trexio_file(
        trexio_file=str(Path(__file__).resolve().parents[1] / "trexio_example_files" / fixture), store_tuple=True
    )
    assert isinstance(leaf, AOs_cart_gto_data)
    assert any(scale != 1.0 for scale in leaf.ao_coefficient_scales)
    points = np.array([[0.27, 0.41, 0.63], [0.91, -0.38, 1.29], [-0.43, 0.67, -0.91]])
    expected_value, expected_grad, expected_lap = _cartesian_trexio_scale_reference(leaf, points)
    actual_value = np.asarray(compute_AOs(leaf, points))
    actual_grad = np.asarray(compute_AOs_grad(leaf, points))
    actual_lap = np.asarray(compute_AOs_laplacian(leaf, points))
    vgl = compute_AOs_value_grad_lap(leaf, points)
    fused_value = np.asarray(vgl[0])
    fused_grad = np.asarray(vgl[1:4])
    fused_lap = np.asarray(vgl[4])

    value_atol, value_rtol = get_tolerance("ao_eval", "strict")
    deriv_atol, deriv_rtol = get_tolerance("ao_grad_lap", "strict")
    np.testing.assert_allclose(actual_value, expected_value, atol=value_atol, rtol=value_rtol, equal_nan=False)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=deriv_atol, rtol=deriv_rtol, equal_nan=False)
    np.testing.assert_allclose(actual_lap, expected_lap, atol=deriv_atol, rtol=deriv_rtol, equal_nan=False)
    np.testing.assert_allclose(fused_value, expected_value, atol=value_atol, rtol=value_rtol, equal_nan=False)
    np.testing.assert_allclose(fused_grad, expected_grad, atol=deriv_atol, rtol=deriv_rtol, equal_nan=False)
    np.testing.assert_allclose(fused_lap, expected_lap, atol=deriv_atol, rtol=deriv_rtol, equal_nan=False)

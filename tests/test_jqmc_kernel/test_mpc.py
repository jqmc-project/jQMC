r"""Tests for the canonical Model Periodic Coulomb (MPC) finite-size correction.

The suite is ordered cheapest and most diagnostic first:

* **Lattice geometry** -- ``L`` and the FFT grid resolved from the cutoff.
  Pure arithmetic, no fixture.
* **Hood-splitting correctness** -- checks the implemented splitting against
  independent numerical quadrature, including zero and nonzero momentum
  points. This is the highest-value check in the file: a sign or factor error
  can shift the correction without being obvious in the final energy.
* **Kernel invariants** -- translation invariance and jit-reusability of
  ``MPC_data``.
* **Interpolation** -- evaluates the tabulated ``V_LR`` between grid nodes
  with the production ``build_mpc_data`` / ``evaluate_v_lr`` and an
  independent reference. The asserted quantity is the mean signed error
  where the electrons sit (around the nuclei), since that biases
  ``<V_MPC - V_Ewald>`` by ``N_e`` times the bias.
* **Physics** -- magnitude and cutoff convergence of the correction.

Detailed derivations of the splitting, interpolation and cell-size
convergence are not reproduced here. The ``1/L^3``
cell-size convergence study needs multiple SCF cells and is not in this
module. The ``physical = no`` bit-identity guarantee is driver-level and is
covered by the MCMC and GFMC driver suites, not here.
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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, vmap
from scipy.integrate import quad

from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_PYTHON_FP64_LOOSE,
    MPC_FIXTURE_CORRECTION_MAGNITUDE_BOUND_HA,
    ATOL_MPC_VLR_INTERPOLATION_MEAN_BIAS,
    ATOL_PBC_TRANSLATION_INVARIANT,
    ATOL_PYTHON_FP64_MEDIUM,
    ATOL_PYTHON_FP64_TIGHT,
    MPC_CUTOFF_HA,
    MPC_GRID_MIN_POINTS,
    MPC_GRID_POINTS_PER_MAX_FREQUENCY,
    RTOL_FD_VS_ANALYTIC_TIGHT,
    RTOL_MPC_GRID_DISCRETIZATION_STABILITY,
    RTOL_PBC_TRANSLATION_INVARIANT,
    RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_TIGHT,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.mpc import (
    MPC_data,
    _fft_int_freqs,
    _hood_g_real,
    _hood_h_G,
    _hood_y,
    build_mpc_data,
    compute_mpc_correction_ee,
    compute_mpc_potential_ee,
    evaluate_v_lr,
    resolve_grid_shape,
    wigner_seitz_radius,
)
from jqmc.trexio_wrapper import read_trexio_file

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# Nothing here reads or writes relative paths -- the only file touched is the
# TREXIO fixture, resolved absolutely from ``__file__``.
pytestmark = pytest.mark.without_chdir

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

# Coarse grid for the fixture-based tests. build_mpc_data runs the N/2N/4N
# FFT extrapolation, so the 4N grid is what sets the runtime -- (12,12,12)
# keeps that at 48^3 and the whole suite in the light tier. Absolute
# accuracy is not what these tests assert.
_TEST_GRID = (12, 12, 12)


class _DuckWF:
    """Minimal wavefunction holder exposing ``geminal_data``."""

    def __init__(self, geminal):
        self.geminal_data = geminal


class _DuckHam:
    """Minimal Hamiltonian holder exposing ``structure_data`` + ``wavefunction_data``."""

    def __init__(self, structure, geminal):
        self.structure_data = structure
        self.wavefunction_data = _DuckWF(geminal)


def _load_fixture(name="lithium.h5"):
    path = os.path.join(_FIXTURE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"TREXIO fixture {name} not present.")
    structure, _aos, _mos_up, _mos_dn, geminal, coulomb = read_trexio_file(trexio_file=path, store_tuple=True)
    if not structure.pbc_flag:
        pytest.skip(f"Fixture {name} is not periodic.")
    return structure, geminal, coulomb


def _random_electrons(structure, geminal, seed=0):
    cell = np.asarray(structure.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_up = int(geminal.num_electron_up)
    n_dn = int(geminal.num_electron_dn)
    frac = rng.uniform(0.1, 0.9, size=(n_up + n_dn, 3))
    r = frac @ cell
    return jnp.asarray(r[:n_up]), jnp.asarray(r[n_up:])


# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------
class TestLatticeGeometry:
    @pytest.mark.light
    def test_wigner_seitz_radius_cubic(self):
        """For a cube the inscribed WS sphere has radius ``a / 2``."""
        a = 15.1178
        assert wigner_seitz_radius(np.eye(3) * a) == pytest.approx(
            a / 2.0, rel=RTOL_PYTHON_FP64_ZERO, abs=ATOL_PYTHON_FP64_TIGHT
        )

    @pytest.mark.light
    def test_wigner_seitz_radius_uses_combinations_not_just_primitives(self):
        """A skewed cell whose shortest lattice vector is ``a1 - a2``.

        ``L = min_{T != 0} |T| / 2`` must search integer combinations, not
        only the three primitive vectors -- otherwise a sheared cell reports
        an ``L`` that is too large and ``g(r)`` is matched to ``1/r`` outside
        the region where that is valid.
        """
        a = 10.0
        cell = np.array([[a, 0.0, 0.0], [a * 0.9, a * 0.1, 0.0], [0.0, 0.0, a]])
        shortest_primitive = min(np.linalg.norm(v) for v in cell)
        # a2 - a1 = (-0.1a, 0.1a, 0) is shorter than any row.
        combo = np.linalg.norm(cell[1] - cell[0])
        assert combo < shortest_primitive
        assert wigner_seitz_radius(cell) == pytest.approx(combo / 2.0, rel=RTOL_PYTHON_FP64_ZERO, abs=ATOL_PYTHON_FP64_TIGHT)

    @pytest.mark.light
    @pytest.mark.parametrize("cutoff_ha", [15.0, 30.0, 60.0])
    def test_resolve_grid_shape_covers_the_cutoff_sphere(self, cutoff_ha):
        """Every ``|G| <= sqrt(2 Ecut)`` must fit on the resolved grid."""
        a = 12.0
        cell = np.eye(3) * a
        shape = resolve_grid_shape(cell, cutoff_ha)
        assert all(n % 2 == 0 for n in shape), "FFT dims should be even"

        b = 2.0 * np.pi / a  # reciprocal spacing for a cube
        g_max_needed = np.sqrt(2.0 * cutoff_ha)
        # Largest integer frequency the grid can represent per axis.
        m_max = min(shape) // 2
        assert m_max * b >= g_max_needed

    @pytest.mark.light
    def test_resolve_grid_shape_is_monotone_in_cutoff(self):
        """Resolve grid shape is monotone in cutoff."""
        cell = np.eye(3) * 12.0
        small = resolve_grid_shape(cell, 15.0)
        large = resolve_grid_shape(cell, 60.0)
        assert all(lo <= hi for lo, hi in zip(small, large))
        assert small != large
        # Keep the non-vacuity check separate from the componentwise bound.
        assert len(small) == len(large) == 3
        assert np.all(np.asarray(small) <= np.asarray(large))

    @pytest.mark.light
    @pytest.mark.parametrize("cutoff_ha", [15.0, 30.0, 60.0, 240.0])
    def test_resolve_grid_shape_oversamples_the_cutoff_sphere_fourfold(self, cutoff_ha):
        r"""The grid must carry ``max(64, 4 m_d)`` points per axis, not ``2 m_d``.

        ``2 m_d + 1`` points already represent the band-limited ``V_LR``; the
        extra factor is for the interpolator between the nodes. Dropping to
        the Nyquist grid gives an MPC correction of the wrong sign on a periodic
        cell (-147 mHa against QMCPACK's +103). A non-cubic cell makes sure the per-axis
        ``m_d`` is used.
        """
        cell = np.diag([12.0, 9.0, 15.0])
        shape = resolve_grid_shape(cell, cutoff_ha)
        g_max = np.sqrt(2.0 * cutoff_ha)
        for n, a in zip(shape, np.diag(cell)):
            m = int(np.ceil(g_max / (2.0 * np.pi / a)))
            assert n >= max(MPC_GRID_MIN_POINTS, MPC_GRID_POINTS_PER_MAX_FREQUENCY * m)


class TestFFTFrequencyIndices:
    """The g_G extrapolation sub-samples finer grids by frequency index.

    Getting those indices even one off silently compares different
    frequencies between grids, which is not a small error: it once cost
    a factor of 60 in the correction at a single cell size, because the
    coarse grid's value at frequency 3 was matched against the fine
    grid's at frequency 2 (opposite signs, so the extrapolation
    diverged instead of merely drifting).
    """

    @pytest.mark.light
    @pytest.mark.parametrize("n", [8, 40, 56, 68, 94, 95, 128, 376])
    def test_matches_numpy_fftfreq_exactly(self, n):
        """Matches numpy fftfreq exactly."""
        expected = np.rint(np.fft.fftfreq(n) * n).astype(int)
        np.testing.assert_array_equal(_fft_int_freqs(n), expected)

    @pytest.mark.light
    @pytest.mark.parametrize("n", [56, 68, 94])
    def test_beats_the_float_round_trip_that_truncation_breaks(self, n):
        """These sizes are exactly where ``fftfreq(n) * n`` is inexact.

        ``fftfreq(94)[3] * 94`` is 2.9999999999999996, so the natural
        ``.astype(int)`` yields 2. The affected sizes are pinned because
        n = 40 happens to be exact, so a check at that size alone cannot
        see the truncation.
        """
        truncated = (np.fft.fftfreq(n) * n).astype(np.int32)
        exact = _fft_int_freqs(n)
        assert not np.array_equal(truncated, exact), f"n={n} no longer exercises the truncation hazard; pick another size"
        np.testing.assert_array_equal(exact, np.rint(np.fft.fftfreq(n) * n).astype(int))


# ---------------------------------------------------------------------------
# Hood g + h splitting
# ---------------------------------------------------------------------------
class TestHoodSplitting:
    @pytest.mark.light
    def test_y_matches_coulomb_in_value_and_slope_at_L(self):
        r"""``g`` must be ``C^1`` at ``r = L``.

        If the value or slope is discontinuous, ``g`` has a kink and its FFT
        no longer converges fast enough for the N/2N/4N extrapolation to be
        meaningful.
        """
        L = 7.5589
        assert float(_hood_y(jnp.asarray(L), L)) == pytest.approx(
            1.0 / L, rel=RTOL_PYTHON_FP64_ZERO, abs=ATOL_PYTHON_FP64_TIGHT
        )

        eps = 1e-6
        dy = (float(_hood_y(jnp.asarray(L), L)) - float(_hood_y(jnp.asarray(L - eps), L))) / eps
        assert dy == pytest.approx(-1.0 / L**2, rel=RTOL_FD_VS_ANALYTIC_TIGHT)

    @pytest.mark.light
    def test_g_is_bare_coulomb_outside_the_sphere(self):
        """G is bare coulomb outside the sphere."""
        L = 5.0
        r = jnp.asarray([L * 1.5, L * 2.0, L * 10.0])
        np.testing.assert_allclose(
            np.asarray(_hood_g_real(r, L)),
            1.0 / np.asarray(r),
            rtol=RTOL_PYTHON_FP64_ZERO,
            atol=ATOL_PYTHON_FP64_MEDIUM,
        )

    @pytest.mark.light
    def test_h_G_zero_limit_is_analytic_value(self):
        """``h_0 = 2 pi L^2 / (5 Omega)`` (CASINO Eq. 194)."""
        L, volume = 7.5589, 15.1178**3
        got = float(_hood_h_G(jnp.asarray(0.0), L, jnp.asarray(volume)))
        assert got == pytest.approx(2.0 * np.pi * L**2 / (5.0 * volume), rel=RTOL_PYTHON_FP64_ZERO, abs=ATOL_PYTHON_FP64_TIGHT)

    @pytest.mark.light
    @pytest.mark.parametrize("G", [0.4156, 0.8312, 1.5, 3.0, 6.0, 10.0])
    def test_h_G_closed_form_matches_quadrature(self, G):
        r"""Regression: the implemented ``h_G`` must match direct numerical quadrature.

        Quadrature is an independent route to the same quantity over the
        tested momentum grid, including zero and nonzero ``G``. An algebra
        slip in the analytic construction would produce a smooth,
        plausible-looking ``f_G`` and a correction of the wrong size, which
        the final energy alone would not reliably catch.
        """
        L, volume = 7.5589, 15.1178**3

        def integrand(r):
            y = -(r * r) / (2.0 * L**3) + 3.0 / (2.0 * L)
            return (1.0 / r - y) * r * np.sin(G * r)

        val, _err = quad(integrand, 0.0, L, limit=400)
        expected = 4.0 * np.pi / (volume * G) * val
        got = float(_hood_h_G(jnp.asarray(G), L, jnp.asarray(volume)))
        assert got == pytest.approx(expected, rel=RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT)

    @pytest.mark.light
    def test_h_G_decays_with_G(self):
        """``h`` is smooth and compact, so ``|h_G|`` must fall off in ``G``."""
        L, volume = 7.5589, 15.1178**3
        gs = jnp.asarray([0.5, 1.0, 2.0, 4.0, 8.0])
        vals = np.abs(np.asarray(_hood_h_G(gs, L, jnp.asarray(volume))))
        assert np.all(np.diff(vals) < 0.0)


# ---------------------------------------------------------------------------
# Kernel invariants
# ---------------------------------------------------------------------------
class TestKernelInvariants:
    @pytest.mark.light
    def test_build_mpc_data_fields_are_finite_and_shaped(self):
        """Build MPC data fields are finite and shaped."""
        structure, geminal, _coulomb = _load_fixture()
        mpc = build_mpc_data(_DuckHam(structure, geminal), grid=_TEST_GRID)

        assert mpc.grid_shape == _TEST_GRID
        assert mpc.v_lr_grid.shape == _TEST_GRID
        assert mpc.v_lr_coef.shape == _TEST_GRID
        assert np.all(np.isfinite(np.asarray(mpc.v_lr_grid)))
        assert np.all(np.isfinite(np.asarray(mpc.v_lr_coef)))
        assert np.isfinite(float(mpc.v_const))
        assert mpc.L > 0.0
        assert mpc.L == pytest.approx(
            wigner_seitz_radius(np.asarray(structure.cell)), rel=RTOL_PYTHON_FP64_ZERO, abs=ATOL_PYTHON_FP64_TIGHT
        )
        assert mpc.f_G_discrepancy >= 0.0

    @pytest.mark.light
    def test_energy_is_invariant_under_lattice_translation(self):
        """Shifting every electron by a lattice vector must not change the energy.

        Exercises the minimum-image pair sum and the periodic wrap of the
        ``V_LR`` interpolation together.
        """
        structure, geminal, _coulomb = _load_fixture()
        mpc = build_mpc_data(_DuckHam(structure, geminal), grid=_TEST_GRID)
        r_up, r_dn = _random_electrons(structure, geminal, seed=3)

        shift = jnp.asarray(np.asarray(structure.cell, dtype=np.float64)[0])
        e0 = float(compute_mpc_potential_ee(mpc, r_up, r_dn))
        e1 = float(compute_mpc_potential_ee(mpc, r_up + shift, r_dn + shift))
        assert e1 == pytest.approx(e0, rel=RTOL_PBC_TRANSLATION_INVARIANT, abs=ATOL_PBC_TRANSLATION_INVARIANT)

    @pytest.mark.light
    def test_mpc_data_survives_repeated_jit_with_distinct_instances(self):
        """Regression: two distinct ``MPC_data`` must both cross a jit boundary.

        ``MPC_data`` once carried the whole ``Structure_data`` as pytree
        *metadata*, which JAX compares with ``==`` when deciding whether a
        cached trace applies. ``Structure_data.__eq__`` raises on the
        ambiguous truth value of its arrays, so the *second* distinct
        instance in a process blew up -- i.e. exactly on checkpoint resume
        and on every wavefunction-optimisation iteration, never in a
        single-shot test.
        """
        structure, geminal, coulomb = _load_fixture()
        ham = _DuckHam(structure, geminal)
        fn = jit(vmap(compute_mpc_potential_ee, in_axes=(None, 0, 0)))

        results = []
        for seed in (11, 12, 13):
            mpc = build_mpc_data(ham, grid=_TEST_GRID)
            r_up, r_dn = _random_electrons(structure, geminal, seed=seed)
            batched = fn(mpc, r_up[None, ...], r_dn[None, ...])
            results.append(float(np.asarray(batched)[0]))

        assert all(np.isfinite(v) for v in results)

    @pytest.mark.light
    def test_mpc_data_is_a_flat_pytree(self):
        """Only arrays are leaves; nothing unhashable sits in the metadata."""
        structure, geminal, _coulomb = _load_fixture()
        mpc = build_mpc_data(_DuckHam(structure, geminal), grid=_TEST_GRID)
        leaves, treedef = jax.tree_util.tree_flatten(mpc)
        assert all(isinstance(x, (jax.Array, np.ndarray)) for x in leaves)
        # Round-tripping requires the metadata to be comparable/hashable.
        assert jax.tree_util.tree_unflatten(treedef, leaves) is not None


# ---------------------------------------------------------------------------
# Interpolation of the tabulated V_LR
# ---------------------------------------------------------------------------
# All-electron H2 in an 11.34 Bohr cube: the density (hence V_LR) is cusped
# at the protons, which is the hardest case for any node-based interpolator
# -- ECP systems are smoother. Ecut = 9 Ha keeps build_mpc_data light; the
# explicit grid is 4 m per axis (m = ceil(sqrt(18) / |b|) = 8), the same
# ratio resolve_grid_shape uses in production, so the node spacing times
# the highest retained wavenumber is the production value pi / 2 and the
# measured bias transfers to any cutoff.
_INTERP_FIXTURE = "H2_pbc_ae_cart.h5"
_INTERP_CUTOFF_HA = 9.0
_INTERP_GRID = (32, 32, 32)
_INTERP_POINTS_PER_NUCLEUS = 300
_INTERP_NUCLEAR_CLOUD_SIGMA_BOHR = 0.25


def _exact_fourier_v_lr(mpc, r_carts):
    """Trigonometric reconstruction of ``V_LR`` from its node values.

    ``build_mpc_data`` masks ``K_G`` to the cutoff sphere, so ``V_LR`` is
    band-limited with ``|k_d| <= m_d < N_d / 2``: the DFT of the node values
    then IS the Fourier series of the field, and evaluating that series at an
    arbitrary point is exact, not another interpolation. Node values are the
    production ``mpc.v_lr_grid``.
    """
    cell = np.asarray(mpc.cell, dtype=np.float64)
    origin = np.asarray(mpc.origin, dtype=np.float64)
    grid = np.asarray(mpc.v_lr_grid, dtype=np.float64)
    coef = np.fft.fftn(grid) / grid.size
    freqs = [np.asarray(_fft_int_freqs(n)) for n in grid.shape]
    frac = np.asarray(r_carts, dtype=np.float64) - origin
    frac = frac @ np.linalg.inv(cell)
    # Separable phases: e^{2 pi i (k1 u + k2 v + k3 w)}.
    p1 = np.exp(2.0j * np.pi * np.outer(frac[:, 0], freqs[0]))
    p2 = np.exp(2.0j * np.pi * np.outer(frac[:, 1], freqs[1]))
    p3 = np.exp(2.0j * np.pi * np.outer(frac[:, 2], freqs[2]))
    val = np.einsum("pa,abc,pb,pc->p", p1, coef, p2, p3)
    return np.real(val)


def _interp_sample_points(structure, seed):
    """Near-nucleus Gaussian clouds and a uniform sample, both in Cartesian Bohr."""
    rng = np.random.default_rng(seed)
    cell = np.asarray(structure.cell, dtype=np.float64)
    nuclei = np.asarray(structure.positions, dtype=np.float64)
    near = np.concatenate(
        [q + _INTERP_NUCLEAR_CLOUD_SIGMA_BOHR * rng.standard_normal((_INTERP_POINTS_PER_NUCLEUS, 3)) for q in nuclei]
    )
    uniform = rng.uniform(0.0, 1.0, size=(near.shape[0], 3)) @ cell
    return near, uniform


@pytest.fixture(scope="module")
def h2_mpc():
    """Production ``MPC_data`` for the interpolation tests, built once per module."""
    structure, geminal, _coulomb = _load_fixture(_INTERP_FIXTURE)
    mpc = build_mpc_data(_DuckHam(structure, geminal), cutoff_ha=_INTERP_CUTOFF_HA, grid=_INTERP_GRID)
    return structure, mpc


class TestInterpolation:
    @pytest.mark.light
    def test_v_lr_spectrum_is_confined_to_the_cutoff_sphere(self, h2_mpc):
        r"""``V_LR`` must be band-limited to the cutoff sphere.

        This is what makes the independent trigonometric reference exact and
        what makes the 4x oversampling meaningful: without the sphere mask the
        corners of the FFT box carry power beyond the cutoff wavenumber and
        the node spacing no longer controls the interpolation error.
        """
        _structure, mpc = h2_mpc
        grid = np.asarray(mpc.v_lr_grid, dtype=np.float64)
        spectrum = np.abs(np.fft.fftn(grid)) / grid.size
        recip = 2.0 * np.pi * np.linalg.inv(np.asarray(mpc.cell, dtype=np.float64)).T
        k = np.stack(np.meshgrid(*[np.asarray(_fft_int_freqs(n)) for n in grid.shape], indexing="ij"), axis=-1)
        g2 = np.sum((k @ recip) ** 2, axis=-1)
        outside = spectrum[0.5 * g2 > _INTERP_CUTOFF_HA]
        assert outside.size > 0
        assert np.max(outside) <= ATOL_PYTHON_FP64_MEDIUM * np.max(spectrum)

    @pytest.mark.light
    def test_evaluate_v_lr_reproduces_the_node_values(self, h2_mpc):
        r"""An interpolating spline must pass through its own nodes.

        If the coefficient construction (including the periodic wrap) is
        wrong, the values at the nodes are already wrong and nothing
        downstream can be right. Detailed spline construction is work in
        progress.
        """
        _structure, mpc = h2_mpc
        cell = np.asarray(mpc.cell, dtype=np.float64)
        origin = np.asarray(mpc.origin, dtype=np.float64)
        n1, n2, n3 = mpc.grid_shape
        idx = np.stack(np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij"), axis=-1).reshape(-1, 3)
        r_nodes = origin + (idx / np.asarray(mpc.grid_shape, dtype=np.float64)) @ cell
        got = np.asarray(evaluate_v_lr(mpc, jnp.asarray(r_nodes)))
        want = np.asarray(mpc.v_lr_grid, dtype=np.float64).reshape(-1)
        scale = np.max(np.abs(want))
        assert np.max(np.abs(got - want)) <= ATOL_PYTHON_FP64_MEDIUM * scale

    @pytest.mark.light
    def test_v_lr_interpolation_bias_near_nuclei_is_negligible(self, h2_mpc):
        r"""Mean signed interpolation error where the electrons are.

        The observable is ``sum_i V_LR(r_i)``, so a per-point bias ``b`` in
        the interpolator shifts ``<V_MPC - V_Ewald>`` by ``N_e b`` while a
        zero-mean scatter averages out; the mean, not the RMS, is therefore
        the figure of merit. The electrons are concentrated at the nuclei,
        where ``V_LR`` has its sharpest feature, so the sample is Gaussian
        clouds around each nucleus; a uniform sample is checked alongside for
        the electrons that are not.

        The reference is an independent reconstruction of the same tabulated
        field, so this is purely an interpolation test and uses only
        production data. A Nyquist-grid trilinear table gives +5.6e-4 Ha on
        this fixture, about 280x the bound; the 4x grid with the tricubic
        B-spline gives +4.1e-7.
        """
        structure, mpc = h2_mpc
        near, uniform = _interp_sample_points(structure, seed=21)

        for label, pts in (("near-nucleus", near), ("uniform", uniform)):
            got = np.asarray(evaluate_v_lr(mpc, jnp.asarray(pts)))
            want = _exact_fourier_v_lr(mpc, pts)
            err = got - want
            assert np.all(np.isfinite(err))
            assert abs(float(np.mean(err))) <= ATOL_MPC_VLR_INTERPOLATION_MEAN_BIAS, (
                f"{label}: mean interpolation bias {np.mean(err):.3e} Ha/point "
                f"(rms {np.sqrt(np.mean(err**2)):.3e}, max {np.max(np.abs(err)):.3e}) "
                f"exceeds {ATOL_MPC_VLR_INTERPOLATION_MEAN_BIAS:.1e}"
            )


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
class TestCorrection:
    @pytest.mark.light
    def test_correction_is_finite_and_small(self):
        """``V_MPC - V_Ewald`` is a finite-size correction, not a leading term.

        The old alpha-Ewald implementation returned ~-10 Ha here (it was
        replacing most of the e-e interaction with a mean field). Anything of
        that magnitude means the kernel is computing the wrong quantity, so
        the bound is deliberately loose but far below it.
        """
        structure, geminal, coulomb = _load_fixture()
        mpc = build_mpc_data(_DuckHam(structure, geminal), grid=_TEST_GRID)
        r_up, r_dn = _random_electrons(structure, geminal, seed=5)

        dV = float(compute_mpc_correction_ee(mpc, coulomb.ewald_data, r_up, r_dn))
        assert np.isfinite(dV)
        assert abs(dV) < MPC_FIXTURE_CORRECTION_MAGNITUDE_BOUND_HA, (
            f"correction {dV} Ha is far too large to be a finite-size effect"
        )

    @pytest.mark.light
    def test_correction_matches_explicit_difference(self):
        """``compute_mpc_correction_ee`` is exactly MPC minus Ewald."""
        from jqmc.coulomb_potential import compute_ewald_potential_ee

        structure, geminal, coulomb = _load_fixture()
        mpc = build_mpc_data(_DuckHam(structure, geminal), grid=_TEST_GRID)
        r_up, r_dn = _random_electrons(structure, geminal, seed=6)

        combined = float(compute_mpc_correction_ee(mpc, coulomb.ewald_data, r_up, r_dn))
        explicit = float(compute_mpc_potential_ee(mpc, r_up, r_dn)) - float(
            compute_ewald_potential_ee(r_up, r_dn, coulomb.ewald_data)
        )
        assert combined == pytest.approx(explicit, rel=RTOL_PYTHON_FP64_ZERO, abs=ATOL_PYTHON_FP64_MEDIUM)

    @pytest.mark.heavy
    def test_correction_converges_in_cutoff(self):
        """``cutoff_ha`` is a convergence knob: increments must shrink.

        A knob without this property (such as a free ``eta``) can move the
        answer by ~40 Ha with no limit. Here successive
        doublings must produce a monotonically shrinking change. Marked heavy:
        the largest cutoff drives a 4N extrapolation grid.
        """
        structure, geminal, coulomb = _load_fixture()
        ham = _DuckHam(structure, geminal)
        r_up, r_dn = _random_electrons(structure, geminal, seed=7)

        vals = []
        for cutoff in (10.0, 20.0, 40.0):
            mpc = build_mpc_data(ham, cutoff_ha=cutoff)
            vals.append(float(compute_mpc_correction_ee(mpc, coulomb.ewald_data, r_up, r_dn)))

        d1 = abs(vals[1] - vals[0])
        d2 = abs(vals[2] - vals[1])
        assert d2 < d1, f"not converging in cutoff: increments {d1:.3e} then {d2:.3e} (values {vals})"

    @pytest.mark.light
    @pytest.mark.parametrize("grid_n", [12, 14, 17])
    def test_correction_is_stable_across_grid_sizes(self, grid_n):
        """Nearby grid sizes must give nearly the same correction.

        The failure this guards against was grid-size-dependent: the
        frequency sub-sampling broke only for certain N, so a single
        grid size proves nothing. 12 is even and exact under the old
        float round-trip, 14 and 17 are not, so the three together
        would have caught it.

        The bound is loose because these are genuinely different
        discretisations of the density -- what matters is that no size
        is wildly out, not that they agree to many digits.
        """
        structure, geminal, coulomb = _load_fixture()
        ham = _DuckHam(structure, geminal)
        r_up, r_dn = _random_electrons(structure, geminal, seed=8)

        ref = float(compute_mpc_correction_ee(build_mpc_data(ham, grid=_TEST_GRID), coulomb.ewald_data, r_up, r_dn))
        got = float(compute_mpc_correction_ee(build_mpc_data(ham, grid=(grid_n,) * 3), coulomb.ewald_data, r_up, r_dn))
        assert got == pytest.approx(ref, rel=RTOL_MPC_GRID_DISCRETIZATION_STABILITY), (
            f"grid {grid_n}^3 gives {got} against {ref} at {_TEST_GRID} -- "
            "a grid-size-dependent discontinuity, not a discretisation difference"
        )

    @pytest.mark.light
    def test_default_cutoff_matches_reference_codes(self):
        """jQMC's default tracks CASINO ``mpc_cutoff`` / QMCPACK ``cutoff``."""
        assert MPC_CUTOFF_HA == 30.0


# ---------------------------------------------------------------------------
# Sampler setup
# ---------------------------------------------------------------------------
class TestSamplerSetup:
    @pytest.mark.light
    @pytest.mark.parametrize("driver", ["mcmc", "gfmc_t", "gfmc_n"])
    def test_sampler_builds_mpc_tables_once_per_hamiltonian(self, driver, monkeypatch):
        """Each production sampler builds MPC tables once per Hamiltonian it holds.

        ``build_mpc_data`` is the most expensive sampler set-up step (a
        real-space density grid of at least 64^3 points). Constructing a
        sampler with ``mpc=True`` must call it exactly once, with the stored
        Hamiltonian and the requested cutoff, and keep that very result.
        Assigning a new Hamiltonian must rebuild it exactly once more. The
        counting wrapper returns real tables built on a coarse grid, so the
        sampler stores a genuine ``MPC_data``; no sampler kernel is mocked.
        """
        import jqmc.jqmc_gfmc as gfmc_module
        import jqmc.jqmc_mcmc as mcmc_module

        from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

        ham = _build_ham_for_cell("H2_pbc_ae_cart.h5", False, None, "SD")
        tables = build_mpc_data(ham, grid=_TEST_GRID)
        calls = []

        def _counting_build(hamiltonian_data, cutoff_ha=MPC_CUTOFF_HA, **kwargs):
            calls.append((hamiltonian_data, cutoff_ha, kwargs))
            return tables

        module = mcmc_module if driver == "mcmc" else gfmc_module
        monkeypatch.setattr(module, "build_mpc_data", _counting_build)
        cutoff = _INTERP_CUTOFF_HA
        if driver == "mcmc":
            sampler = mcmc_module.MCMC(hamiltonian_data=ham, mcmc_seed=1, num_walkers=1, mpc=True, mpc_cutoff=cutoff)
        else:
            cls = gfmc_module.GFMC_t if driver == "gfmc_t" else gfmc_module.GFMC_n
            nproj = {} if driver == "gfmc_t" else {"num_mcmc_per_measurement": 16}
            sampler = cls(hamiltonian_data=ham, num_walkers=1, mcmc_seed=1, alat=0.30, mpc=True, mpc_cutoff=cutoff, **nproj)

        def _stored_tables():
            return sampler._MCMC__mpc_data if driver == "mcmc" else sampler._mpc_data

        assert len(calls) == 1, f"{driver}: construction built MPC tables {len(calls)} times, expected once"
        held, got_cutoff, extra = calls[0]
        assert held is sampler.hamiltonian_data
        # The sampler's own sampling comm is forwarded, so the tables are
        # built once per comm (rank 0) and broadcast, not once per rank.
        assert got_cutoff == cutoff and extra == {"comm": sampler._mpi_comm}
        assert _stored_tables() is tables

        sampler.hamiltonian_data = ham
        assert len(calls) == 2, f"{driver}: a new Hamiltonian must rebuild MPC tables once; got {len(calls) - 1} builds"
        assert calls[1][0] is sampler.hamiltonian_data
        assert calls[1][1] == cutoff and calls[1][2] == {"comm": sampler._mpi_comm}
        assert _stored_tables() is tables


# ---------------------------------------------------------------------------
# Shared (build-once) set-up
# ---------------------------------------------------------------------------
def _former_eager_min_image_r(cell, n_fft, n_max):
    """Eager reference for ``_cell_grid_min_image_r``: one Python iteration per image per chunk.

    It keeps the chunking over ``jqmc.mpc._MIN_IMAGE_CHUNK``
    nodes (the ``(chunk, 3) @ (3, 3)`` products round differently for
    different row counts), as the reference that the compiled image sweep
    must reproduce bit for bit.
    """
    import jqmc.mpc as mpc_module
    from jqmc.structure import _inv_3x3

    n1, n2, n3 = (int(x) for x in n_fft)
    inv_cell = _inv_3x3(cell)
    n_nodes = n1 * n2 * n3
    rng = np.arange(-n_max, n_max + 1, dtype=np.float64)
    ia, ib, ic = np.meshgrid(rng, rng, rng, indexing="ij")
    T_images = jnp.asarray(np.stack([ia.ravel(), ib.ravel(), ic.ravel()], axis=-1), dtype=jnp.float64) @ cell
    pieces = []
    for start in range(0, n_nodes, mpc_module._MIN_IMAGE_CHUNK):
        stop = min(start + mpc_module._MIN_IMAGE_CHUNK, n_nodes)
        flat = jnp.arange(start, stop, dtype=jnp.int64)
        frac = jnp.stack(
            [
                (flat // (n2 * n3)).astype(jnp.float64) / n1,
                ((flat // n3) % n2).astype(jnp.float64) / n2,
                (flat % n3).astype(jnp.float64) / n3,
            ],
            axis=-1,
        )
        disp = frac @ cell
        disp_frac = disp @ inv_cell
        disp = (disp_frac - jnp.round(disp_frac)) @ cell
        best = jnp.full((disp.shape[0],), jnp.inf, dtype=jnp.float64)
        for t in range(T_images.shape[0]):
            d = disp + T_images[t]
            best = jnp.minimum(best, jnp.sum(d * d, axis=-1))
        pieces.append(jnp.sqrt(best))
    return jnp.concatenate(pieces, axis=0).reshape(n1, n2, n3)


def _assert_same_bits(actual, reference, label):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    assert actual.dtype == reference.dtype and actual.shape == reference.shape, label
    assert np.all(np.isfinite(reference)), f"{label}: reference is not finite"
    assert actual.tobytes() == reference.tobytes(), f"{label}: not bitwise identical"


def _assert_same_mpc_data(actual, reference):
    for name in ("cell", "origin", "v_lr_grid", "v_lr_coef", "v_const"):
        _assert_same_bits(getattr(actual, name), getattr(reference, name), name)
    for name in ("n_max", "grid_shape", "L", "cutoff_ha", "f_G_discrepancy"):
        assert getattr(actual, name) == getattr(reference, name), name


def _assert_close_mpc_data(actual, reference, rtol, atol):
    for name in ("cell", "origin", "v_lr_grid", "v_lr_coef", "v_const"):
        a = np.asarray(getattr(actual, name))
        r = np.asarray(getattr(reference, name))
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(r)), f"{name}: not finite"
        np.testing.assert_allclose(a, r, rtol=rtol, atol=atol, err_msg=name)
    for name in ("n_max", "grid_shape", "L", "cutoff_ha"):
        assert getattr(actual, name) == getattr(reference, name), name
    assert np.isfinite(actual.f_G_discrepancy)
    np.testing.assert_allclose(actual.f_G_discrepancy, reference.f_G_discrepancy, rtol=rtol, atol=atol)


def _shared_setup_ham():
    """The periodic lithium fixture as a duck Hamiltonian; a missing fixture is an error, not a skip."""
    structure, _aos, _mos_up, _mos_dn, geminal, _coulomb = read_trexio_file(
        trexio_file=os.path.join(_FIXTURE_DIR, "lithium.h5"), store_tuple=True
    )
    assert structure.pbc_flag
    return _DuckHam(structure, geminal)


class _DoneRequest:
    """A completed non-blocking request (rank 0 of the replay has already finished)."""

    def Test(self):
        return True

    def Wait(self):
        return None


class _ReplayComm:
    """In-process stand-in for one rank of a two-rank communicator.

    Rank 0 records what it broadcasts on a shared channel; rank 1, run
    afterwards, receives exactly those payloads. This exercises the real
    broadcast protocol of ``build_mpc_data`` without launching MPI.
    """

    def __init__(self, rank, channel):
        self._rank = rank
        self._channel = channel

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return 2

    def bcast(self, obj, root=0):
        assert root == 0
        if self._rank == 0:
            self._channel.append(("bcast", obj))
            return obj
        kind, payload = self._channel.pop(0)
        assert kind == "bcast"
        return payload

    def Ibarrier(self):
        return _DoneRequest()

    def Bcast(self, buf, root=0):
        assert root == 0
        if self._rank == 0:
            self._channel.append(("Bcast", np.array(buf, copy=True)))
            return
        kind, payload = self._channel.pop(0)
        assert kind == "Bcast" and payload.shape == buf.shape and payload.dtype == buf.dtype
        buf[...] = payload


class TestSharedSetup:
    _CELLS = {
        # Diamond-Si fcc primitive cell of jqmc-example07 (n_max = 3 there).
        "fcc-si": (np.array([[0.0, 5.08430814, 5.08430814], [5.08430814, 0.0, 5.08430814], [5.08430814, 5.08430814, 0.0]]), 3),
        "triclinic": (np.array([[7.0, 0.0, 0.0], [2.1, 6.0, 0.0], [-1.3, 0.7, 8.2]]), 2),
    }

    @pytest.mark.light
    @pytest.mark.gdd_electron_configuration(
        mode="not-required", reason="Minimum-image distances of fixed FFT grid nodes; no electron is placed."
    )
    @pytest.mark.parametrize("cell_name", ["fcc-si", "triclinic"])
    def test_compiled_min_image_sweep_matches_the_former_eager_sweep(self, cell_name, monkeypatch):
        """The compiled image sweep reproduces the eager per-image reference loop to the fp64 ulp level.

        The chunk size is lowered so the grid spans several chunks with a
        short final chunk (10 x 12 x 14 = 1680 nodes, chunk 500), so the
        chunk bookkeeping is exercised, not just one full chunk.

        Bitwise equality is what the production chunk sizes give (measured on
        the 64^3/128^3/256^3 grids of jqmc-example07), but XLA rounds
        ``sum(d * d)`` of a fused kernel and of an eager op differently for
        some array lengths (here: 180 and 500 rows differ by one ulp), so the
        verdict is the central fp64 ulp-level relative bound with zero atol:
        a wrong image, wrapping or chunk offset is O(1) off.
        """
        import jqmc.mpc as mpc_module

        cell_np, n_max = self._CELLS[cell_name]
        cell = jnp.asarray(cell_np, dtype=jnp.float64)
        n_fft = (10, 12, 14)
        monkeypatch.setattr(mpc_module, "_MIN_IMAGE_CHUNK", 500)
        actual = mpc_module._cell_grid_min_image_r(cell, n_fft, n_max)
        reference = np.asarray(_former_eager_min_image_r(cell, n_fft, n_max))
        actual = np.asarray(actual)
        assert actual.shape == reference.shape == n_fft
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(actual, reference, rtol=RTOL_PYTHON_FP64_TIGHT, atol=ATOL_BIT_EXACT)

    @pytest.mark.light
    @pytest.mark.gdd_electron_configuration(
        mode="not-required", reason="Minimum-image distances of fixed FFT grid nodes; no electron is placed."
    )
    @pytest.mark.parametrize("cell_name", ["fcc-si", "triclinic"])
    def test_image_pruning_is_bitwise_exact(self, cell_name, monkeypatch):
        """Dropping the images with ``|T| > 2 R`` leaves every bit of ``|r|`` unchanged.

        Same compiled sweep on both sides; the reference keeps the whole
        ``(2 n_max + 1)^3`` box (margin so large that nothing is pruned).
        The pruned run must actually drop images, or the check is vacuous.
        """
        import jqmc.mpc as mpc_module

        cell_np, n_max = self._CELLS[cell_name]
        cell = jnp.asarray(cell_np, dtype=jnp.float64)
        n_fft = (16, 18, 20)
        seen = []
        original_sweep = mpc_module._min_image_r_sweep

        def _recording_sweep(disp, T_images):
            seen.append(int(T_images.shape[0]))
            return original_sweep(disp, T_images)

        monkeypatch.setattr(mpc_module, "_min_image_r_sweep", _recording_sweep)
        pruned = mpc_module._cell_grid_min_image_r(cell, n_fft, n_max)
        n_pruned = seen[-1]
        monkeypatch.setattr(mpc_module, "_MIN_IMAGE_PRUNE_MARGIN", 1.0e6)
        full_box = mpc_module._cell_grid_min_image_r(cell, n_fft, n_max)
        assert seen[-1] == (2 * n_max + 1) ** 3
        assert n_pruned < seen[-1], f"{cell_name}: pruning kept every image"
        _assert_same_bits(pruned, full_box, f"{cell_name} pruned vs full image box")

    @pytest.mark.light
    def test_build_mpc_data_matches_the_former_eager_build(self, monkeypatch):
        """Every ``MPC_data`` field matches the build that uses the eager per-image reference sweep.

        Bitwise on this fixture; asserted with the central fp64
        accumulator bound because a one-ulp difference in ``|r|`` (see the
        sweep test above) propagates through three FFTs and the
        extrapolation.
        """
        import jqmc.mpc as mpc_module

        ham = _shared_setup_ham()
        actual = build_mpc_data(ham, grid=_TEST_GRID)
        monkeypatch.setattr(mpc_module, "_cell_grid_min_image_r", _former_eager_min_image_r)
        reference = build_mpc_data(ham, grid=_TEST_GRID)
        _assert_close_mpc_data(actual, reference, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_comm_build_runs_once_on_rank_zero_and_broadcasts_bitwise(self, monkeypatch):
        """With a multi-rank comm only rank 0 builds; every rank holds rank 0's tables bit for bit.

        Rank 1 is given no Hamiltonian at all and the local builder is made
        to fail there, so any table it returns can only have come from the
        broadcast. Both ranks must equal a plain local build bitwise.
        """
        import jqmc.mpc as mpc_module

        ham = _shared_setup_ham()
        reference = build_mpc_data(ham, grid=_TEST_GRID)

        channel = []
        built_on_rank0 = build_mpc_data(ham, grid=_TEST_GRID, comm=_ReplayComm(0, channel))
        _assert_same_mpc_data(built_on_rank0, reference)

        def _must_not_build(*args, **kwargs):
            raise AssertionError("a non-root rank rebuilt the MPC tables")

        monkeypatch.setattr(mpc_module, "_build_mpc_data_local", _must_not_build)
        received_on_rank1 = build_mpc_data(None, grid=_TEST_GRID, comm=_ReplayComm(1, channel))
        assert channel == [], "rank 1 did not consume every broadcast"
        _assert_same_mpc_data(received_on_rank1, reference)

    @pytest.mark.light
    @pytest.mark.gdd_electron_configuration(
        mode="not-required", reason="The rank-0 build is replaced by a failing stub before any grid point is evaluated."
    )
    def test_comm_build_failure_on_rank_zero_is_raised_on_every_rank(self, monkeypatch):
        """A rank-0 build failure is re-raised on rank 0 and reported on the others (no hang)."""
        import jqmc.mpc as mpc_module

        def _failing_build(*args, **kwargs):
            raise ValueError("synthetic rank-0 MPC failure")

        monkeypatch.setattr(mpc_module, "_build_mpc_data_local", _failing_build)
        channel = []
        with pytest.raises(ValueError, match="synthetic rank-0 MPC failure"):
            build_mpc_data(None, comm=_ReplayComm(0, channel))
        with pytest.raises(RuntimeError, match="synthetic rank-0 MPC failure"):
            build_mpc_data(None, comm=_ReplayComm(1, channel))
        assert channel == []

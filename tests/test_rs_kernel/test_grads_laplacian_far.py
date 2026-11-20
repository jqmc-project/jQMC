"""Far-from-home periodic AO grad/Laplacian + kinetic equivalence, Rust vs jqmc.

Regression guard for native periodic sphe-pGTO **image folding**:
jqmc wraps every electron coordinate into the home cell
(``_wrap_into_cell_if_pbc``) before evaluating the AO, because its truncated
image sum is enumerated relative to the home cell. A native real-Gamma pGTO
grad/lap path that evaluates at the RAW coordinate puts an electron several
lattice vectors outside the home cell beyond the static image tiling, and the
sum misses the dominant images -- grad/lap and the derived kinetic energy
collapse (native T -> ~0 vs jqmc's periodic-invariant value).

``test_grads_laplacian.py`` / ``test_mcmc_sweep_pbc.py`` only exercise
NEAR-nucleus electrons (inside the home cell, where the wrap is a no-op), so
they cannot see it. This module shifts the SAME near configuration by
integer lattice vectors ``T`` and asserts the Rust grad/lap/kinetic are
unchanged (as they must be -- the periodic wavefunction is translationally
invariant under ``T``), i.e. still bit-close to jqmc at ALL configs.
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

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

import jax.numpy as jnp

from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.determinant import compute_geminal_all_elements, compute_grads_and_laplacian_ln_Det
from jqmc.hamiltonians import compute_local_energy_fast
from jqmc.wavefunction import compute_kinetic_energy

from tests.test_rs_kernel._rs_fixtures import energy_with_ecp_dtn_ratio_omitted, rs_energy_ham, rs_ham
from tests.test_rs_kernel.test_mcmc_observables import _WALKER_COND_MAX, _walker_batch

_ZONES_LNDET = ("ao_eval", "ao_grad_lap", "det_grad_lap")
_ZONES_KINETIC = ("wf_kinetic", "ao_eval", "ao_grad_lap", "det_grad_lap", "jastrow_grad_lap")

# Periodic real-Gamma sphe-pGTO, jnone + j12 (kinetic touches the AO grad/lap
# regardless of the Jastrow channel; j12 additionally sweeps the periodic
# Jastrow grad/lap at the same far configs).
_PBC_PATTERNS = ["pbc_gamma_sphepgto_jnone", "pbc_gamma_sphepgto_j12"]

# Integer lattice-vector shifts. ``(0,0,0)`` is the NEAR control (inside the home
# cell); the rest push electrons well beyond the static image tiling.
_SHIFTS = [(0, 0, 0), (1, 2, 0), (2, 3, -1), (4, -3, 2)]


def _near_config(ham, seed=0):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    r_up = R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.4, size=(nu, 3))
    r_dn = R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.4, size=(nd, 3))
    return r_up, r_dn


def _lattice_shift(ham, n):
    cell = np.asarray(ham.structure_data.cell, float)  # rows = lattice vectors
    return n[0] * cell[0] + n[1] * cell[1] + n[2] * cell[2]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PBC_PATTERNS)
@pytest.mark.parametrize("shift", _SHIFTS)
def test_grads_laplacian_ln_det_far(pattern, shift):
    """Grads laplacian ln det far."""
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn = _near_config(ham)
    T = _lattice_shift(ham, shift)
    r_up, r_dn = r_up + T, r_dn + T
    gu, gd, lu, ld = jqmc_rs.compute_grads_and_laplacian_ln_det(gem, r_up, r_dn)
    Gu, Gd, Lu, Ld = (np.asarray(x, float) for x in compute_grads_and_laplacian_ln_Det(gem, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNDET, level="strict")
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol), (shift, np.max(np.abs(gu - Gu)))
    assert np.allclose(gd, Gd, atol=atol, rtol=rtol), (shift, np.max(np.abs(gd - Gd)))
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol), (shift, np.max(np.abs(lu - Lu)))
    assert np.allclose(ld, Ld, atol=atol, rtol=rtol), (shift, np.max(np.abs(ld - Ld)))


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PBC_PATTERNS)
@pytest.mark.parametrize("shift", _SHIFTS)
def test_kinetic_energy_far(pattern, shift):
    """Kinetic energy far."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    r_up, r_dn = _near_config(ham, seed=2)
    T = _lattice_shift(ham, shift)
    r_up, r_dn = r_up + T, r_dn + T
    k_rust = jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)
    k_jqmc = float(compute_kinetic_energy(wf, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    assert np.isclose(k_rust, k_jqmc, rtol=rtol, atol=atol), (shift, k_rust, k_jqmc)


# Native AGGREGATE e_L (``compute_local_energy_fast``, the production observable) at far
# configs, closing the coverage gap that hid the far-image e_L bug: the
# earlier rs e_L tests sampled only NEAR-nucleus electrons, so the reused-inverse kinetic (SINGLE
# AO convention) AND the ECP non-local base/mesh AO -- both of which enumerate the periodic image
# tiling relative to the home cell -- went untested at a far coordinate. ``j123`` exercises the J3
# grad/lap + non-local J3 mesh too. ``compute_local_energy_fast`` reuses a sweep-consistent SM
# inverse; native must match jqmc's ``compute_local_energy_fast`` (which wraps every AO eval
# internally while keeping the walker unwrapped) at the SAME far coords and inverse.
_EL_PATTERNS = ["pbc_gamma_sphepgto_jnone", "pbc_gamma_sphepgto_j12", "pbc_gamma_sphepgto_j123"]
_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")
_MESH = icosahedron_sym_mesh_Nv12


@pytest.mark.light
@pytest.mark.parametrize("pattern", _EL_PATTERNS)
@pytest.mark.parametrize("shift", _SHIFTS)
def test_local_energy_fast_far(pattern, shift):
    """Native ``compute_local_energy_fast`` == jqmc's at a lattice-shifted (far) config, reusing a
    sweep-consistent SM inverse. Regression guard for the far-image e_L: without the
    determinant AO / ECP-non-local base-orbital home-cell wrap the native periodic e_L diverges
    up to ~1 Ha at electrons wandered several cells out during the walk."""
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn = _near_config(ham, seed=1)
    T = _lattice_shift(ham, shift)
    r_up, r_dn = r_up + T, r_dn + T
    RT = np.eye(3)
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float) @ RT
    # sweep-consistent inverse: the full-image jqmc geminal is what the sweep's running SM inverse
    # tracks (bit-close over the chain); jqmc wraps the AO internally so this is translation-stable.
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gem, r_up, r_dn)))
    native = float(jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, w, gp, g_inv))
    jax_eL = float(compute_local_energy_fast(ham, r_up, r_dn, jnp.asarray(RT), jnp.asarray(g_inv), None))
    # MEDIUM level: in full precision medium resolves
    # to the same values as strict, so the fp64 judgment is unchanged; only the
    # mixed envelope widens. The aggregate e_L = T + V cancels strongly on these
    # fixtures (measured T = -4.44 against e_L = -0.164, a 27x cancellation), so
    # both backends' fp32 noise scales with |T| while a strict rtol scales with
    # |e_L|: the measured mixed parity residual, 1.1e-6 absolute, is an excellent
    # 2.6e-7 of T yet 7e-6 of e_L -- 2.3x over strict, unreachable without
    # bit-exact f32 alignment of the entire kinetic pipeline. fp64 parity on the
    # same battery is 2e-14. Mixed precision uses this
    # medium envelope; full precision keeps the strict pair.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium" if is_mixed_precision_enabled() else "strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (shift, native, jax_eL, abs(native - jax_eL))


# Native COMPLEX-twist (k != Gamma) aggregate e_L (``compute_local_energy_fast_complex``, the
# production observable) at far configs. Unlike the real periodic path, the complex batch
# AO (``ComplexOrb::value`` / ``value_grad_lap``) home-cell-wraps INTERNALLY (its ``WrapCtx``), so
# the kinetic grad^2 G stays convention-consistent with the reused COMPLEX SM inverse far from the
# home cell WITHOUT a ``via_single`` split; the ECP non-local (complex det ratio, NN=1) is
# kept complex exactly as jqmc's VMC forward, so Im(e_L) == Im(T + V_nl).
# The supplemental B-spline cell exercises active DTN at nonzero twist without
# broadening the unrelated force/parameter fixture matrices.
_EL_COMPLEX_PATTERNS = ["twist_sphepgto_jnone", "twist_sphepgto_j12", "bspline_twist_jdtn"]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _EL_COMPLEX_PATTERNS)
@pytest.mark.parametrize("shift", _SHIFTS)
def test_local_energy_fast_complex_far(pattern, shift):
    """Native ``compute_local_energy_fast_complex`` == jqmc's complex ``compute_local_energy_fast``
    at a lattice-shifted (far) config, reusing a sweep-consistent COMPLEX SM inverse. Regression
    guard for the complex-twist e_L: the internally-wrapping complex batch AO must keep the
    reused-inverse kinetic translation-stable, and the unprojected complex ECP non-local must
    match jqmc's forward energy, for electrons wandered several cells out during the walk.
    Includes committed QE Li B-spline with nonzero twist and active DTN, on the
    first seed-0 conditioned walker and all four integer lattice translations.
    Its actual translated positions must have finite cond(G) <= 64. Full and
    mixed use separate same-mode comparisons at the existing medium tolerance.
    Its DTN contribution to the nonlocal ECP must be detectable at the energy
    comparison tolerance; kinetic and the other Jastrow channels stay intact
    in this fault control. Existing pGTO rows retain their seed-1 positions;
    their one-up/one-down scalar geminal has cond(G)=1 and satisfies the same
    explicit finite-condition check without adding a configuration screen."""
    ham = rs_energy_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    if pattern == "bspline_twist_jdtn":
        up_batch, dn_batch = _walker_batch(ham, nw=1, seed=0)
        r_up, r_dn = up_batch[0], dn_batch[0]
    else:
        r_up, r_dn = _near_config(ham, seed=1)
    T = _lattice_shift(ham, shift)
    r_up, r_dn = r_up + T, r_dn + T
    geminal_matrix = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn))
    cond_g = float(np.linalg.cond(geminal_matrix))
    assert np.isfinite(cond_g) and cond_g <= _WALKER_COND_MAX
    RT = np.eye(3)
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float) @ RT
    # Complex full-image geminal inverse (the running COMPLEX SM inverse the sweep tracks; jqmc
    # wraps the AO internally so it is translation-stable). Plain inverse == G^{-1} for full rank.
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gem, r_up, r_dn))).astype(np.complex128)
    re, im = jqmc_rs.compute_local_energy_fast_complex(ham, r_up, r_dn, w, gp, g_inv)
    native = complex(re, im)
    jax_eL = complex(compute_local_energy_fast(ham, r_up, r_dn, jnp.asarray(RT), jnp.asarray(g_inv), None))
    # MEDIUM level: in full precision medium resolves
    # to the same values as strict, so the fp64 judgment is unchanged; only the
    # mixed envelope widens. The aggregate e_L = T + V cancels strongly on these
    # fixtures (measured T = -4.44 against e_L = -0.164, a 27x cancellation), so
    # both backends' fp32 noise scales with |T| while a strict rtol scales with
    # |e_L|: the measured mixed parity residual, 1.1e-6 absolute, is an excellent
    # 2.6e-7 of T yet 7e-6 of e_L -- 2.3x over strict, unreachable without
    # bit-exact f32 alignment of the entire kinetic pipeline. fp64 parity on the
    # same battery is 2e-14. Mixed precision uses this
    # medium envelope; full precision keeps the strict pair.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium" if is_mixed_precision_enabled() else "strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (shift, native, jax_eL, abs(native - jax_eL))
    if pattern == "bspline_twist_jdtn":
        omitted = energy_with_ecp_dtn_ratio_omitted(ham, r_up, r_dn, jax_eL)
        assert np.isfinite(jax_eL) and np.isfinite(omitted)
        assert not np.isclose(omitted, jax_eL, rtol=rtol, atol=atol, equal_nan=False)

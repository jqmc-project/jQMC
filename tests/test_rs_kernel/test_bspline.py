"""B-spline orbital basis (`AOs_bspline_data`, periodic real-Gamma) vs jqmc.

Tensor-product 3D cardinal (uniform) B-spline on a regular grid -- the jQMC einspline
analog. Two layers verified:
  1. The AO leaf value + Cartesian gradient + Laplacian (`bspline::AosBsplineData`,
     `compute_aos_bspline_vgl`) vs jqmc `_compute_AOs_value_grad_lap`, including
     electrons OUTSIDE the home cell (exercises the modular-wrap gather).
  2. The full geminal: ln|det| and kinetic energy through the SD bspline geminal
     (OrbData routes to the spline leaf; MO layer = identity) vs jqmc.

Large B-spline systems (48+48 electrons, grid 60x60x54, quintic order-5, oblique
cell, ~75 MB coeffs/spin) are not committed, so this file uses the much smaller committed B-spline pattern
``bspline_gamma_jnone`` (QE Li_gamma projection, real-Gamma, ECP, 1+1 electrons)
via ``tests.test_rs_kernel._rs_fixtures.rs_ham`` instead. A Rust==XLA equivalence is
code-path sensitive, not system-size sensitive, so the small fixture exercises the
same bspline leaf / geminal / kinetic code paths a large system does.

DTN+bspline coverage: there is no committed bspline+DTN *fixture*, but
``_build_ham_for_cell`` (the same builder ``_rs_fixtures.py`` uses) accepts
``spec={"dtn": True}`` on the ``QEPROJ:bspline:*`` sentinel directly, and (verified)
builds and runs cleanly on this small system. On a 96-electron DTN B-spline system jqmc's
``compute_local_energy`` triggers a JAX XLA-accumulation segfault in a load-accumulated
xdist worker; that heavy-system caveat does not apply at this system size, so
``test_bspline_native_vmc`` below covers BOTH jnone and jdtn.
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

import pathlib

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc.hamiltonians import compute_local_energy  # noqa: E402
from jqmc.determinant import compute_ln_det_geminal_all_elements  # noqa: E402
from jqmc.wavefunction import compute_kinetic_energy  # noqa: E402
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12  # noqa: E402
from jqmc.derivatives.ao import bspline_value_grad_lap_dcell  # noqa: E402

import jqmc_rs  # noqa: E402

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = pytest.mark.with_chdir

_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, float)
_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, float)

# AO value/grad/lap zones: the fused compute_AOs_value_grad_lap-equivalent kernel.
# The AO leaf rows (value/grad/lap and their d/dcell) use the proportional bound
# ``atol + rtol * max|f|``: the zone atol is the only absolute floor. The B-spline
# control points carry the Omega^-1/2 cell normalization, so AO magnitudes
# are O(1e-2); a unit offset inside the relative term would dominate and hide relative
# errors up to about 1/max|f| rtol.
_ZONE_AO = ("ao_eval", "ao_grad_lap")
# Gradient / Laplacian rows: ``_compute_AOs_value_grad_lap`` (real and complex) evaluates
# every component in ``ao_grad_lap`` and casts only the value to ``ao_eval``, so grad/lap
# cross ``ao_grad_lap`` alone (fp64 in both modes); the Rust kernel is fp64. The value row
# keeps the ``ao_eval`` pair. Every d/dcell row, dval included, comes from the fp64
# ``bspline_value_grad_lap_dcell`` kernel with no ``ao_eval`` cast, so all five resolve on
# ``ao_grad_lap``.
_ZONE_AO_GRAD_LAP = ("ao_grad_lap",)
# Per-row zones for the (val, gx, gy, gz, lap) tuple and its d/dcell twin.
_ZONES_VGL = (_ZONE_AO, _ZONE_AO_GRAD_LAP, _ZONE_AO_GRAD_LAP, _ZONE_AO_GRAD_LAP, _ZONE_AO_GRAD_LAP)
_ZONES_VGL_DCELL = (_ZONE_AO_GRAD_LAP,) * 5
# ln|det| / kinetic energy zones: AO feeds the determinant + kinetic assembly.
_ZONE_DET = ("ao_eval", "det_eval")
_ZONE_KIN = ("ao_eval", "det_eval", "wf_kinetic")
# Full local-energy zones (kinetic + Ewald + ECP Coulomb).
_ZONE_EL = ("ao_eval", "det_eval", "coulomb", "ewald", "wf_kinetic")


def _cell(gd):
    return np.asarray(gd.orb_data_up_spin.aos_data.structure_data.cell, float)


def _dtn_ham():
    """The same committed QE B-spline stem, with an active DTN Jastrow (verified to build
    and run cleanly at this system size; see module docstring)."""
    return _build_ham_for_cell("QEPROJ:bspline:Li_gamma:ECP", True, {"dtn": True}, "SD")


@pytest.mark.light
def test_bspline_python_extraction_propagates_type_errors():
    """The native reader accepts a real spline but rejects a malformed complex flag.

    A deterministic synthetic cubic spline supplies every valid native input
    field. Only its Python ``_complex`` flag is replaced. The normal proxy
    must evaluate first, so missing fields cannot masquerade as flag refusal.
    """
    from tests.test_jqmc_kernel.test_atomic_orbital import _build_bspline_leaf, _nongto_cubic_structure

    source = pathlib.Path(__file__).resolve().parents[2] / "jqmc" / "rs" / "src" / "bspline.rs"
    assert ".extract().unwrap_or(false)" not in source.read_text()
    aos = _build_bspline_leaf(_nongto_cubic_structure())
    positions = np.asarray([[0.13, 0.27, -0.19]], dtype=np.float64)

    class FlagProxy:
        def __init__(self, flag):
            self._complex = flag

        def __getattr__(self, name):
            return getattr(aos, name)

    values = jqmc_rs.compute_aos_bspline_vgl(FlagProxy(False), positions)
    assert len(values) == 5
    for value in values:
        assert value.shape == (aos.num_ao, 1)
        assert np.all(np.isfinite(value))
    with pytest.raises(TypeError, match="bool|Bool|boolean"):
        jqmc_rs.compute_aos_bspline_vgl(FlagProxy("not-a-boolean"), positions)


@pytest.mark.light
def test_bspline_ao_value_grad_lap():
    """AO leaf value/grad/lap vs jqmc `_compute_AOs_value_grad_lap`, electrons spanning
    [-0.3, 1.4) in fractional coords to exercise the periodic modular-wrap gather."""
    ham = rs_ham("bspline_gamma_jnone")
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert type(aos).__name__ == "AOs_bspline_data"
    cell = np.asarray(aos.structure_data.cell, float)
    rng = np.random.default_rng(0)
    r = (rng.random((8, 3)) * 1.7 - 0.3) @ cell
    ref = [np.asarray(x) for x in aos._compute_AOs_value_grad_lap(r)]
    got = [np.asarray(x) for x in jqmc_rs.compute_aos_bspline_vgl(aos, r)]
    for name, zones, g, f in zip(("val", "gx", "gy", "gz", "lap"), _ZONES_VGL, got, ref):
        atol, rtol = get_tolerance_xla_rust_consistency(zones)
        d = np.abs(g - f).max()
        assert d <= atol + rtol * np.abs(f).max(), (name, d)


@pytest.mark.light
def test_bspline_complex_ao_value_grad_lap():
    """COMPLEX (twist k != Gamma) B-spline AO leaf value/grad/lap vs jqmc
    `_compute_AOs_value_grad_lap`. Exercises the Bloch-phase product rule
    ``psi = u e^{ik.r}`` (grad `(grad u + i k u) e^{ik.r}`, lap
    `(lap u + 2 i k . grad u - |k|^2 u) e^{ik.r}`) that the native complex sweep
    layers on top of the reused REAL spline `u`. Electrons span several cells so
    the Bloch phase across cell boundaries (uses the FULL r) is exercised
    alongside the cell-periodic spline wrap of `u`."""
    ham = rs_ham("bspline_twist_jnone")
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert type(aos).__name__ == "AOs_bspline_complex_data"
    cell = np.asarray(aos.structure_data.cell, float)
    rng = np.random.default_rng(1)
    r = (rng.random((8, 3)) * 1.7 - 0.3) @ cell + rng.integers(-1, 2, size=(8, 3)) @ cell
    ref = [np.asarray(x) for x in aos._compute_AOs_value_grad_lap(r)]
    got = [np.asarray(x) for x in jqmc_rs.compute_aos_bspline_complex_vgl(aos, np.asarray(r, dtype=np.float64))]
    for name, zones, g, f in zip(("val", "gx", "gy", "gz", "lap"), _ZONES_VGL, got, ref):
        atol, rtol = get_tolerance_xla_rust_consistency(zones)
        assert np.iscomplexobj(np.asarray(g)), f"{name} not complex"
        d = np.abs(np.asarray(g) - np.asarray(f)).max()
        assert d <= atol + rtol * np.abs(np.asarray(f)).max(), (name, d)


@pytest.mark.light
def test_tiled_bspline_native_vgl_and_implicit_identity_determinant():
    """Native tiled B-splines use the primitive grid, per-band k, and no dense identity MO."""
    from jqmc.atomic_orbital import AOs_bspline_tiled_complex_data
    from jqmc.determinant import SD_data
    from jqmc.molecular_orbital import MOs_data
    from jqmc.structure import tile_structure_data
    from tests.test_jqmc_kernel.test_atomic_orbital import _build_bspline_leaf, _nongto_cubic_structure

    primitive = _nongto_cubic_structure()
    supercell = tile_structure_data(primitive, (2, 1, 1))
    base = _build_bspline_leaf(primitive)
    coefficients = np.asarray(base.spline_coefficients, dtype=np.complex128)
    blocks = (coefficients, (0.65 - 0.15j) * coefficients + 0.1j * np.roll(coefficients, 1, axis=1))
    aos = AOs_bspline_tiled_complex_data.build(
        structure_data=supercell,
        spline_coefficients_by_k=blocks,
        spline_shape=tuple(base.spline_shape),
        spline_order=base.spline_order,
        tiling_matrix=(2, 1, 1),
        k_points=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
    )
    positions = np.array(
        [[0.31, 0.72, 1.13], [3.43, 1.27, 4.91], [-1.17, 6.32, -2.04], [5.11, 3.08, 0.47]],
        dtype=np.float64,
    )
    reference = [np.asarray(value) for value in aos._compute_AOs_value_grad_lap(positions)]
    native = [np.asarray(value) for value in jqmc_rs.compute_aos_bspline_complex_vgl(aos, positions)]
    for name, zones, got, expected in zip(("val", "gx", "gy", "gz", "lap"), _ZONES_VGL, native, reference):
        atol, rtol = get_tolerance_xla_rust_consistency(zones)
        error = np.abs(got - expected).max()
        assert error <= atol + rtol * np.abs(expected).max(), (name, error)

    mos = MOs_data.build_identity(aos_data=aos)
    assert np.asarray(mos.mo_coefficients).size == 0
    determinant = SD_data.build(num_electron_up=4, num_electron_dn=4, mos_up=mos)
    up = positions
    down = positions + np.array([[0.17, -0.09, 0.13]], dtype=np.float64)
    expected_ln_det = float(compute_ln_det_geminal_all_elements(determinant, up, down))
    got_ln_det = float(jqmc_rs.compute_ln_det_complex(determinant, up, down))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE_DET)
    assert abs(got_ln_det - expected_ln_det) <= atol + rtol * abs(expected_ln_det)


@pytest.mark.light
def test_bspline_geminal_lndet_and_kinetic():
    """Full SD bspline geminal ln|det| + kinetic energy vs jqmc."""
    ham = rs_ham("bspline_gamma_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    cell = _cell(gd)
    atol_d, rtol_d = get_tolerance_xla_rust_consistency(_ZONE_DET)
    atol_t, rtol_t = get_tolerance_xla_rust_consistency(_ZONE_KIN)
    for seed in range(2):
        rng = np.random.default_rng(seed)
        ru = rng.random((gd.num_electron_up, 3)) @ cell
        rd = rng.random((gd.num_electron_dn, 3)) @ cell
        lref = float(compute_ln_det_geminal_all_elements(gd, ru, rd))
        lgot = float(jqmc_rs.compute_ln_det_geminal_all_elements(gd, ru, rd))
        assert abs(lgot - lref) <= atol_d + rtol_d * (abs(lref) + 1.0), (seed, abs(lgot - lref))
        tref = float(compute_kinetic_energy(wf, ru, rd))
        tgot = float(jqmc_rs.compute_kinetic_energy(gd, wf.jastrow_data, ru, rd))
        assert abs(tgot - tref) <= atol_t + rtol_t * (abs(tref) + 1.0), (seed, abs(tgot - tref))


@pytest.mark.light
def test_bspline_native_vmc():
    """The full native VMC chain (`native_vmc`) drives a B-spline geminal end-to-end (the real
    sweep + real e_L already handle bspline via compute_single / orb_value_grad_lap). Smoke +
    sanity: it runs, yields finite per-block energies in a physical range, and the per-config e_L
    equals jQMC `compute_local_energy` (spot-check via the deterministic Rust
    `compute_local_energy`). Both jnone and jdtn are exercised (see module docstring for why the
    jdtn e_L cross-check is safe to run at this small system size)."""
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE_EL)
    for kind, ham in (("jnone", rs_ham("bspline_gamma_jnone")), ("jdtn", _dtn_ham())):
        en, acc = jqmc_rs.native_vmc(ham, _W, _GP, 42, 1.0, 0.0, 30, 1, 3)
        assert len(en) == 3 and np.all(np.isfinite(en)) and 0.0 <= acc <= 1.0, (kind, en, acc)
        assert np.all(np.asarray(en) < 0), (kind, en)  # Li ground-state e_L is negative

        gd = ham.wavefunction_data.geminal_data
        cell = _cell(gd)
        rng = np.random.default_rng(0)
        ru = rng.random((gd.num_electron_up, 3)) @ cell
        rd = rng.random((gd.num_electron_dn, 3)) @ cell
        egot = float(jqmc_rs.compute_local_energy(ham, ru, rd, _W, _GP))
        eref = float(compute_local_energy(ham, ru, rd, jnp.eye(3)))
        assert abs(egot - eref) <= atol + rtol * (abs(eref) + 1.0), (kind, egot, eref)


@pytest.mark.light
def test_bspline_ao_stress_dcell():
    """B-spline AO STRESS building block: directional d(val,grad,lap)/dcell along dcell vs
    jqmc `bspline_value_grad_lap_dcell`. The spline FORCE
    (d/dR_nuc) is identically 0 -- control points have no nuclear-position dependence -- so
    only this cell channel (via inv_cell/metric) is non-trivial."""
    ham = rs_ham("bspline_gamma_jnone")
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    cell = np.asarray(aos.structure_data.cell, float)
    rng = np.random.default_rng(3)
    r = (rng.random((6, 3)) * 1.4 - 0.2) @ cell
    for _ in range(2):
        dcell = rng.normal(size=(3, 3)) * 0.3
        ref = [np.asarray(x) for x in bspline_value_grad_lap_dcell(aos, r, dcell)]
        got = [np.asarray(x) for x in jqmc_rs.compute_aos_bspline_vgl_dcell(aos, r, dcell)]
        for name, zones, g, f in zip(("dval", "dgx", "dgy", "dgz", "dlap"), _ZONES_VGL_DCELL, got, ref):
            atol, rtol = get_tolerance_xla_rust_consistency(zones)
            d = np.abs(g - f).max()
            assert d <= atol + rtol * np.abs(f).max(), (name, d)


# NOTE: a full-`compute_local_energy` (kinetic+Ewald+ECP) smoke check on both jnone/jdtn is
# covered directly by test_bspline_native_vmc above (no heavy-system segfault risk at this
# committed system's small size). bspline correctness is additionally covered by the AO
# value/grad/lap + geminal ln|det|/kinetic tests above.

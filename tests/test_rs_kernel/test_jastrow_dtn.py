"""Rust==XLA parity for the DTN polynomial Jastrow (J1/J2/J3 family) and for
J1/J2 on skewed cells.

Closes three holes in the Jastrow parity suite:

  * HOLE 1: ``jqmc_rs.compute_dtn_value`` was only exercised inside aggregate
    tests (``test_mcmc_observables`` / ``test_periodic_invariants``), and
    ``jqmc_rs.compute_dtn_ratio`` / ``jqmc_rs.compute_grads_and_laplacian_dtn``
    were never invoked directly in the parity suite. The XLA physics battery
    (``tests/test_jqmc_kernel/test_jastrow_dtn.py``) validates DTN
    value/grad/lap/ratio/cusp densely against numpy oracles / FD, but nothing
    pinned the Rust kernels to the XLA production functions.

  * HOLE 2: no anisotropic / oblique cell appeared in ANY Jastrow parity test
    (all periodic parity fixtures are cubic lithium). The Rust min-image
    reduction is ``round(x @ inv_cell) @ cell`` (``jastrow_factor.rs``), exact
    only under an inscribed-radius condition; skewed cells are the untested
    failure class. This module runs the DTN family AND J1/J2 on the battery's
    oblique / triclinic cells, with electrons displaced by integer lattice
    combinations so the min-image path is live.

  * HOLE 3: coincidence-limit / cusp-region sampling was absent from parity.
    This module probes electron-electron and electron-nucleus separations at
    ``~1e-6`` (the coincidence limit) on molecular + oblique-PBC systems.

All comparisons are Rust-kernel-vs-XLA-production-function; the Kato cusp
coefficients themselves stay XLA-only (built by ``Jastrow_DTN_data.build``).

The battery helpers (``_build_dtn`` / ``_draw_electrons`` /
``_cell_fixtures_for_dist`` / ``_h2_structure`` / ``_h2_pbc_structure``) and the
triclinic structure builder from ``test_jastrow.py`` are imported directly
rather than re-encoding their constants.
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

import sys

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

import jax
import jax.numpy as jnp

import jqmc.jastrow_factor as jastrow_factor_module
from jqmc._precision import (
    get_dtype_np,
    get_tolerance,
    get_tolerance_min,
    get_tolerance_xla_rust_consistency,
    precision_policy_snapshot,
)
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_two_body_data,
    _compute_ratio_Jastrow_part_rank1_update,
    _init_jastrow_dtn_ratio_state,
    compute_grads_and_laplacian_Jastrow_DTN,
    compute_Jastrow_DTN,
    compute_Jastrow_one_body,
    compute_Jastrow_part,
    compute_Jastrow_two_body,
)

# Import the XLA battery helpers so the oblique cell / DTN builders are shared
# rather than re-encoded here.
from tests.conftest import make_pbc_structure
from tests.test_jqmc_kernel.test_jastrow import TestJastrowPBCAnalytic as _JastrowPBCAnalytic
from tests.test_jqmc_kernel.test_jastrow_dtn import (
    _CUSP_PROBE_EXPECTED,
    _CUSP_PROBE_LABELS,
    _CUSP_PROBE_SHAPES,
    _CUSP_PROBE_SMOOTHNESS,
    _build_dtn,
    _cell_fixtures_for_dist,
    _draw_electrons,
    _h2_pbc_structure,
    _cusp_probe_dtn,
    _cusp_probe_slopes,
    _h2_structure,
    _legacy_casino_sign_bias,
)

# Zone tolerances (no inline numeric literals in asserts -- pre-commit policy).
_ZONES_VALUE = ("jastrow_eval",)
_ZONES_GRAD_LAP = ("jastrow_grad_lap",)
# Ratio crosses both the value zone (recompute reference) and the ratio zone.
_ZONES_RATIO = ("jastrow_eval", "jastrow_ratio")

# Coincidence-limit / cusp-region electron offsets. These are INPUT literals
# (the geometry sampled), not assert-bound tolerances, so they are allowed.
_CUSP_OFFSET = 1.0e-6  # e-e / e-n separation probed at coincidence
_FAR_FROM_NUCLEI = 5.0  # place e-e coincidence pair away from any atom


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------


def _oblique_pbc_structure():
    """Two H atoms in the battery's oblique (triclinic) ~10 Bohr cell.

    Mirrors the oblique entry of the battery's ``_cell_fixtures_for_dist``
    (cell ``[[10,0,0],[3,9.5,0],[1.5,2,11]]``, WS inscribed radius ~4.67 Bohr,
    so the default DTN cutoff ``L=3`` sits safely inside it). This is the
    skewed-cell case that no prior Jastrow parity test covered.
    """
    _, cell_obl, _ = _cell_fixtures_for_dist()[1]
    cell = np.asarray(cell_obl, dtype=np.float64)
    return make_pbc_structure(
        cell,
        positions=np.array([[2.0, 2.0, 2.0], [4.4, 2.0, 2.0]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )


_STRUCT_KEYS = ("molecular", "cubic", "oblique")


def _structure(key):
    if key == "molecular":
        return _h2_structure()
    if key == "cubic":
        return _h2_pbc_structure()
    if key == "oblique":
        return _oblique_pbc_structure()
    raise ValueError(key)


def _cell_of(sd) -> np.ndarray:
    return np.array([sd.vec_a, sd.vec_b, sd.vec_c], dtype=np.float64)


def _displace_by_lattice(sd, r_up, r_dn, seed):
    """Shift each electron by a random integer lattice combination.

    For a periodic system every DTN / J1 / J2 evaluation depends only on the
    minimum image, so the shifted configuration must give an identical value
    (and grad / lap / ratio) in BOTH the XLA and the Rust path. This makes the
    ``round(x @ inv_cell) @ cell`` reduction live and is the direct probe of
    HOLE 2 on the oblique cell. Molecular systems pass through unchanged.
    """
    if not sd.pbc_flag:
        return r_up.copy(), r_dn.copy()
    cell = _cell_of(sd)
    rng = np.random.default_rng(seed + 9001)

    def _shift(r):
        combos = rng.integers(-2, 3, size=(r.shape[0], 3))
        return (r + combos @ cell).astype(np.float64)

    return _shift(r_up), _shift(r_dn)


# ---------------------------------------------------------------------------
# 1. Value parity: Rust compute_dtn_value vs XLA compute_Jastrow_DTN
# ---------------------------------------------------------------------------


class TestDTNValueParity:
    @pytest.mark.light
    @pytest.mark.parametrize("key", _STRUCT_KEYS)
    @pytest.mark.parametrize("preset", ["u", "u_chi", "u_chi_f"])
    @pytest.mark.parametrize("seed", [11, 23, 41])
    def test_dtn_value_parity(self, key, preset, seed):
        """Rust DTN value matches XLA to the ``jastrow_eval`` floor, and the
        periodic value is min-image invariant under integer lattice shifts
        (both paths)."""
        atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
        sd = _structure(key)
        data = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=7)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=seed)

        J_xla = float(compute_Jastrow_DTN(data, r_up, r_dn))
        J_rust = float(jqmc_rs.compute_dtn_value(data, r_up, r_dn))
        assert np.isfinite(J_xla) and np.isfinite(J_rust)
        assert np.isclose(J_rust, J_xla, atol=atol, rtol=rtol), (key, preset, J_rust, J_xla)

        if sd.pbc_flag:
            dr_up, dr_dn = _displace_by_lattice(sd, r_up, r_dn, seed)
            J_xla_d = float(compute_Jastrow_DTN(data, dr_up, dr_dn))
            J_rust_d = float(jqmc_rs.compute_dtn_value(data, dr_up, dr_dn))
            # Parity on the displaced (min-image-live) configuration.
            assert np.isclose(J_rust_d, J_xla_d, atol=atol, rtol=rtol), (key, "disp", J_rust_d, J_xla_d)
            # Lattice-translation invariance in each path independently: this
            # is what fails if the skewed-cell round() reduction is wrong.
            assert np.isclose(J_xla_d, J_xla, atol=atol, rtol=rtol), (key, "xla-invariance", J_xla_d, J_xla)
            assert np.isclose(J_rust_d, J_rust, atol=atol, rtol=rtol), (key, "rust-invariance", J_rust_d, J_rust)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# 2. Grad / Laplacian parity: Rust vs XLA analytic production path
# ---------------------------------------------------------------------------


class TestDTNGradLaplacianParity:
    @pytest.mark.light
    @pytest.mark.parametrize("key", _STRUCT_KEYS)
    @pytest.mark.parametrize("preset", ["u", "u_chi_f"])
    @pytest.mark.parametrize("seed", [4, 19])
    def test_dtn_grads_laplacian_parity(self, key, preset, seed):
        """Rust per-electron grads / Laplacians match the XLA analytic
        production kernel ``compute_grads_and_laplacian_Jastrow_DTN``.

        Rust vs XLA uses the ``independent_fp32`` pair: in mixed precision the
        two backends are separate fp32 implementations of the zone, so their
        difference is bounded by twice the single-route strict pair. The XLA
        lattice-translation invariance compares one route with itself and keeps
        the strict pair."""
        pair_atol, pair_rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="independent_fp32")
        atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
        sd = _structure(key)
        data = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=7)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=seed)

        g_up_x, g_dn_x, l_up_x, l_dn_x = compute_grads_and_laplacian_Jastrow_DTN(data, r_up, r_dn)
        g_up_r, g_dn_r, l_up_r, l_dn_r = jqmc_rs.compute_grads_and_laplacian_dtn(data, r_up, r_dn)

        for label, x, r in (
            ("grad_up", g_up_x, g_up_r),
            ("grad_dn", g_dn_x, g_dn_r),
            ("lap_up", l_up_x, l_up_r),
            ("lap_dn", l_dn_x, l_dn_r),
        ):
            xa = np.asarray(x, dtype=np.float64)
            ra = np.asarray(r, dtype=np.float64)
            assert np.all(np.isfinite(xa)) and np.all(np.isfinite(ra)), (key, preset, label)
            np.testing.assert_allclose(ra, xa, atol=pair_atol, rtol=pair_rtol, err_msg=f"{key}/{preset}/{label}")

        if sd.pbc_flag:
            # Grad / lap are min-image invariant too: displaced parity + XLA
            # invariance is the skewed-cell probe for the derivative kernels.
            dr_up, dr_dn = _displace_by_lattice(sd, r_up, r_dn, seed)
            g_up_xd, g_dn_xd, l_up_xd, l_dn_xd = compute_grads_and_laplacian_Jastrow_DTN(data, dr_up, dr_dn)
            g_up_rd, g_dn_rd, l_up_rd, l_dn_rd = jqmc_rs.compute_grads_and_laplacian_dtn(data, dr_up, dr_dn)
            for label, x, r in (
                ("grad_up_disp", g_up_xd, g_up_rd),
                ("lap_up_disp", l_up_xd, l_up_rd),
            ):
                np.testing.assert_allclose(
                    np.asarray(r), np.asarray(x), atol=pair_atol, rtol=pair_rtol, err_msg=f"{key}/{label}"
                )
            # invariance in XLA (grad/lap unchanged by a lattice translation).
            np.testing.assert_allclose(np.asarray(g_up_xd), np.asarray(g_up_x), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(l_up_xd), np.asarray(l_up_x), atol=atol, rtol=rtol)
            del g_dn_xd, l_dn_xd, g_dn_rd, l_dn_rd
        jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="fixed near-coincidence coordinates and their lattice images",
)
def test_periodic_dtn_grad_lap_preserves_close_displacements_in_lattice_images(backend):
    """The same short e-e and e-N displacements survive in different lattice images."""
    cell = np.diag([8.0, 8.0, 8.0])
    structure = make_pbc_structure(
        cell,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )
    data = _build_dtn(structure, preset="u_chi_f", seed=7)
    separation = 2.0**-24
    r_up = np.array([[separation, 0.0, 0.0]])
    r_dn = np.array([[0.0, separation, 0.0]])
    evaluate = {
        "xla": compute_grads_and_laplacian_Jastrow_DTN,
        "rust": jqmc_rs.compute_grads_and_laplacian_dtn,
    }[backend]
    canonical = evaluate(data, r_up, r_dn)
    translated = evaluate(data, r_up + cell[0], r_dn - cell[1])
    atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
    for actual, reference in zip(translated, canonical, strict=True):
        assert np.all(np.isfinite(reference)) and np.any(np.asarray(reference) != 0)
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# 3. Single-move ratio parity: Rust compute_dtn_ratio vs XLA
# ---------------------------------------------------------------------------


def _xla_rank1_dtn_ratio(data, r_up, r_dn, new_up, new_dn):
    """XLA production single-move ratio via the rank-1 DTN update path."""
    jd = Jastrow_data(jastrow_dtn_data=data)
    ru = jnp.asarray(r_up)
    rd = jnp.asarray(r_dn)
    state = _init_jastrow_dtn_ratio_state(data, ru, rd)
    out = _compute_ratio_Jastrow_part_rank1_update(
        jd, ru, rd, jnp.asarray(new_up)[None], jnp.asarray(new_dn)[None], dtn_state=state
    )
    return float(np.asarray(out)[0])


class TestDTNRatioParity:
    @pytest.mark.light
    @pytest.mark.parametrize("key", _STRUCT_KEYS)
    @pytest.mark.parametrize("is_up", [True, False])
    @pytest.mark.parametrize("seed", [5, 27])
    def test_dtn_ratio_parity(self, key, is_up, seed):
        """Rust single-move DTN ratio ``exp(J(R') - J(R))`` matches (a) the
        full XLA recompute and (b) the XLA rank-1 update path, including moves
        that cross the cell boundary on the oblique cell."""
        atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_RATIO, level="strict")
        sd = _structure(key)
        data = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=7)
        r_up, r_dn = _draw_electrons(sd, 2, 1, seed=seed)

        rng = np.random.default_rng(seed + 500)
        idx = 0
        base = r_up[idx] if is_up else r_dn[idx]
        # A move that on a periodic cell can push the electron out of the home
        # cell so the min-image branch selects a different image for the new
        # position than for the old one.
        if sd.pbc_flag:
            step = 0.4 * _cell_of(sd)[0] + rng.normal(scale=0.3, size=3)
        else:
            step = rng.normal(scale=0.6, size=3)
        new = (base + step).astype(np.float64)

        new_up = r_up.copy()
        new_dn = r_dn.copy()
        if is_up:
            new_up[idx] = new
        else:
            new_dn[idx] = new

        truth = float(np.exp(compute_Jastrow_DTN(data, new_up, new_dn) - compute_Jastrow_DTN(data, r_up, r_dn)))
        rank1 = _xla_rank1_dtn_ratio(data, r_up, r_dn, new_up, new_dn)
        rust = float(jqmc_rs.compute_dtn_ratio(data, r_up, r_dn, is_up, idx, new.tolist()))

        assert np.isfinite(rust)
        assert np.isclose(rust, truth, atol=atol, rtol=rtol), (key, is_up, "truth", rust, truth)
        assert np.isclose(rust, rank1, atol=atol, rtol=rtol), (key, is_up, "rank1", rust, rank1)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# 4. Coincidence-limit / cusp-region parity (HOLE 3)
# ---------------------------------------------------------------------------


def _assert_empty_dtn_derivative(actual, reference, component):
    """The absent down-spin channel has the exact component shape, not numerical evidence."""
    assert component in ("grad_dn", "lap_dn")
    expected_shape = (0, 3) if component == "grad_dn" else (0,)
    assert actual.shape == reference.shape == expected_shape


class TestDTNCoincidenceCuspRegion:
    @pytest.mark.light
    @pytest.mark.parametrize("key", ["molecular", "oblique"])
    def test_dtn_coincidence_and_cusp_region_parity(self, key):
        """Value / grad-lap / ratio parity when electrons are ``~1e-6`` apart
        (parallel + antiparallel e-e) and when an electron is ``~1e-6`` from a
        nucleus. Both kernels must stay finite and agree; the Kato coefficients
        stay XLA-only."""
        av, rv = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
        ag, rg = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
        ar, rr = get_tolerance_xla_rust_consistency(_ZONES_RATIO, level="strict")
        sd = _structure(key)
        data = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=7)

        # A point safely away from both H nuclei (H1@origin-ish, H2 nearby) for
        # the e-e coincidence pair; in the oblique cell it stays inside.
        base = np.array([_FAR_FROM_NUCLEI, _FAR_FROM_NUCLEI, _FAR_FROM_NUCLEI])
        eps = np.array([_CUSP_OFFSET, 0.0, 0.0])

        # --- e-e coincidence, parallel (both up) ---
        r_up_par = np.stack([base, base + eps], axis=0)
        r_dn_par = np.zeros((0, 3))
        # --- e-e coincidence, antiparallel (one up, one dn) ---
        r_up_anti = base.reshape(1, 3)
        r_dn_anti = (base + eps).reshape(1, 3)
        # --- e-n coincidence: an electron ~1e-6 from nucleus 0 ---
        R0 = np.asarray(sd.positions, dtype=np.float64)[0]
        r_up_en = (R0 + eps).reshape(1, 3)
        r_dn_en = (base).reshape(1, 3)

        for label, ru, rd in (
            ("ee_parallel", r_up_par, r_dn_par),
            ("ee_antiparallel", r_up_anti, r_dn_anti),
            ("en_coincidence", r_up_en, r_dn_en),
        ):
            J_x = float(compute_Jastrow_DTN(data, ru, rd))
            J_r = float(jqmc_rs.compute_dtn_value(data, ru, rd))
            assert np.isfinite(J_x) and np.isfinite(J_r), (key, label, "value-finite")
            assert np.isclose(J_r, J_x, atol=av, rtol=rv), (key, label, "value", J_r, J_x)

            g_up_x, g_dn_x, l_up_x, l_dn_x = compute_grads_and_laplacian_Jastrow_DTN(data, ru, rd)
            g_up_r, g_dn_r, l_up_r, l_dn_r = jqmc_rs.compute_grads_and_laplacian_dtn(data, ru, rd)
            for gl_label, x, r in (
                ("grad_up", g_up_x, g_up_r),
                ("grad_dn", g_dn_x, g_dn_r),
                ("lap_up", l_up_x, l_up_r),
                ("lap_dn", l_dn_x, l_dn_r),
            ):
                xa = np.asarray(x, dtype=np.float64)
                ra = np.asarray(r, dtype=np.float64)
                assert np.all(np.isfinite(xa)) and np.all(np.isfinite(ra)), (key, label, gl_label, "finite")
                if label == "ee_parallel" and gl_label in ("grad_dn", "lap_dn"):
                    _assert_empty_dtn_derivative(ra, xa, gl_label)
                    continue
                comparison_label = f"{label}/{gl_label}"
                np.testing.assert_allclose(ra, xa, atol=ag, rtol=rg, err_msg=f"{key}/{comparison_label}")

            # Ratio parity: move up-electron 0 to within ~1e-6 of a partner.
            if ru.shape[0] >= 1:
                new = (base + 2.0 * eps).astype(np.float64)
                new_up = ru.copy()
                new_up[0] = new
                truth = float(np.exp(compute_Jastrow_DTN(data, new_up, rd) - compute_Jastrow_DTN(data, ru, rd)))
                rust = float(jqmc_rs.compute_dtn_ratio(data, ru, rd, True, 0, new.tolist()))
                assert np.isfinite(rust), (key, label, "ratio-finite")
                assert np.isclose(rust, truth, atol=ar, rtol=rr), (key, label, "ratio", rust, truth)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# 4b. Kato cusp slopes of the Rust gradient kernel for every smoothness C
# ---------------------------------------------------------------------------


class TestDTNRustCuspSlopeEverySmoothness:
    """The Rust DTN gradient kernel carries the Kato cusps for even AND odd C.

    The Rust kernel consumes jQMC's inflated coefficients, so a wrong cusp
    algebra reaches it unchanged and backend parity alone cannot see it.  This
    measures the physical slope of the Rust gradient directly, with the same
    explicit coalescence probes and Richardson extrapolation as the XLA check.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("shape", sorted(_CUSP_PROBE_SHAPES))
    @pytest.mark.parametrize("C", _CUSP_PROBE_SMOOTHNESS)
    @pytest.mark.parametrize("geometry", ["molecular", "periodic"])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="explicit coalescence probes; the seed initializes DTN coefficients only",
    )
    def test_rust_gradient_has_kato_cusps_for_every_smoothness(self, geometry, C, shape):
        """The Rust gradient gives e-e slopes 1/2 and 1/4, AE e-n slope -Z and ECP e-n slope 0.

        Judged at the active ``jastrow_grad_lap`` precision (fp64 full, fp32 mixed).
        """
        data = _cusp_probe_dtn(geometry, C, shape)
        slopes = _cusp_probe_slopes(data, jqmc_rs.compute_grads_and_laplacian_dtn)
        atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
        np.testing.assert_allclose(slopes, _CUSP_PROBE_EXPECTED, atol=atol, rtol=rtol, err_msg=str(_CUSP_PROBE_LABELS))

    @pytest.mark.light
    @pytest.mark.parametrize("C", [3, 5])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="explicit",
        reason="explicit coalescence probes; the seed initializes DTN coefficients only",
    )
    def test_rust_cusp_verdict_rejects_legacy_odd_smoothness_sign(self, monkeypatch, C):
        """The same verdict fails on odd-C biases multiplied by (-1)^C after a passing baseline."""
        self.test_rust_gradient_has_kato_cusps_for_every_smoothness("molecular", C, "L1.5-low-order")
        module = sys.modules[__name__]
        original = module._cusp_probe_dtn
        monkeypatch.setattr(module, "_cusp_probe_dtn", lambda *args: _legacy_casino_sign_bias(original(*args)))
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            self.test_rust_gradient_has_kato_cusps_for_every_smoothness("molecular", C, "L1.5-low-order")


# ---------------------------------------------------------------------------
# 4c. Broad XLA==Rust sampling
# ---------------------------------------------------------------------------

# Electron counts (balanced, imbalanced, closed) and the placements sampled for
# each.  The named offsets are INPUT geometry (fractions of the cutoff or Bohr
# separations), not assert-bound tolerances.
_BROAD_ELECTRON_COUNTS = ((3, 2), (4, 1), (2, 2))
_BROAD_PLACEMENTS = ("random", "near-cutoff", "ee-parallel", "ee-antiparallel", "en-coalescence")
_BROAD_CUTOFF_GAP = 1.0e-3  # electron at (1 - gap) * L from the all-electron nucleus
_BROAD_COALESCENCE = 1.0e-3  # e-e / e-n separation in Bohr


def _broad_configuration(data, n_up, n_dn, placement, seed):
    """Seeded DTN configuration with one deliberately placed electron.

    ``random`` keeps the seeded draw; the other placements move one electron
    to ``(1 - gap) L`` from nucleus 0 (inside every channel's support), next to
    a same-spin or opposite-spin partner, or next to nucleus 0.
    """
    sd = data.structure_data
    r_up, r_dn = _draw_electrons(sd, n_up, n_dn, seed=seed)
    nuclei = np.asarray(sd.positions, dtype=np.float64)
    if not sd.pbc_flag:
        r_up = r_up + nuclei[0] + np.array([1.0, 0.0, 0.0])
        r_dn = r_dn + nuclei[0] + np.array([1.5, 0.0, 0.0])
    rng = np.random.default_rng(seed + 77)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    cutoff = min(
        float(np.asarray(data.L_u).reshape(())), float(np.min(np.asarray(data.L_chi))), float(np.min(np.asarray(data.L_f)))
    )
    r_up, r_dn = r_up.copy(), r_dn.copy()
    if placement == "near-cutoff":
        r_up[0] = nuclei[0] + (1.0 - _BROAD_CUTOFF_GAP) * cutoff * direction
    elif placement == "ee-parallel":
        r_up[1] = r_up[0] + _BROAD_COALESCENCE * direction
    elif placement == "ee-antiparallel":
        r_dn[0] = r_up[0] + _BROAD_COALESCENCE * direction
    elif placement == "en-coalescence":
        r_up[0] = nuclei[0] + _BROAD_COALESCENCE * direction
    elif placement != "random":
        raise ValueError(placement)
    return r_up, r_dn


class TestDTNBroadParity:
    """XLA==Rust DTN value, gradient and ratio over the sampling the
    narrow fixtures above leave out.

    The narrow parity tests use H2 (Z=1), C=4, one cutoff/order shape and
    two up plus one down electron at seeded positions.  This battery adds odd
    and even C, two (L, order) shapes, an all-electron and a pseudized species,
    spin-imbalanced and same-spin-pair electron sets, and electrons placed
    1e-3 L inside the cutoff and 1e-3 Bohr from a partner or the nucleus,
    for molecules and a periodic cell, at the strict backend-consistency pair
    of each zone in each precision mode.

    The Laplacian is deliberately not asserted here: at this sampling the
    mixed-mode Laplacian disagreement is set by fp32 conditioning, not by a
    backend defect (both kernels run the declared fp32 zone; on a 432-configuration
    superset of this battery each backend's own fp32-vs-fp64 Laplacian error
    exceeds the strict pair on about half), so this comparison needs a separately
    reviewed mixed-precision policy.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("shape", sorted(_CUSP_PROBE_SHAPES))
    @pytest.mark.parametrize("C", [3, 4, 5])
    @pytest.mark.parametrize("geometry", ["molecular", "periodic"])
    @pytest.mark.gdd_electron_configuration(
        mode="required",
        position_source="seeded-random",
        reason="seeded draws plus named near-cutoff and coalescence placements of one electron",
    )
    def test_dtn_broad_value_gradient_ratio_parity(self, geometry, C, shape):
        """Rust DTN value and single-move ratio equal XLA at the strict pair, the gradient at the independent-fp32 pair.

        The mixed gradient is two separate fp32 implementations of the zone,
        whose difference is bounded by twice the single-route strict pair.
        """
        av, rv = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
        ag, rg = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="independent_fp32")
        ar, rr = get_tolerance_xla_rust_consistency(_ZONES_RATIO, level="strict")
        data = _cusp_probe_dtn(geometry, C, shape)
        step = np.array([0.3, -0.2, 0.1])
        for n_up, n_dn in _BROAD_ELECTRON_COUNTS:
            for placement in _BROAD_PLACEMENTS:
                r_up, r_dn = _broad_configuration(data, n_up, n_dn, placement, seed=n_up + 10 * n_dn)
                label = f"{geometry}/C{C}/{shape}/{n_up}u{n_dn}d/{placement}"

                J_xla = float(compute_Jastrow_DTN(data, r_up, r_dn))
                J_rust = float(jqmc_rs.compute_dtn_value(data, r_up, r_dn))
                assert np.isfinite(J_xla) and np.isfinite(J_rust), label
                assert np.isclose(J_rust, J_xla, atol=av, rtol=rv), (label, "value", J_rust, J_xla)

                grads_xla = compute_grads_and_laplacian_Jastrow_DTN(data, r_up, r_dn)[:2]
                grads_rust = jqmc_rs.compute_grads_and_laplacian_dtn(data, r_up, r_dn)[:2]
                for spin, x, r in zip(("up", "dn"), grads_xla, grads_rust):
                    xa = np.asarray(x, dtype=np.float64)
                    ra = np.asarray(r, dtype=np.float64)
                    assert np.all(np.isfinite(xa)) and np.all(np.isfinite(ra)), (label, spin)
                    np.testing.assert_allclose(ra, xa, atol=ag, rtol=rg, err_msg=f"{label}/grad_{spin}")

                new = r_up[0] + step
                new_up = r_up.copy()
                new_up[0] = new
                truth = float(np.exp(compute_Jastrow_DTN(data, new_up, r_dn) - compute_Jastrow_DTN(data, r_up, r_dn)))
                rust = float(jqmc_rs.compute_dtn_ratio(data, r_up, r_dn, True, 0, new.tolist()))
                assert np.isfinite(rust), label
                assert np.isclose(rust, truth, atol=ar, rtol=rr), (label, "ratio", rust, truth)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# 4d. Zone-dtype conformance of every DTN kernel entry point
# ---------------------------------------------------------------------------

# entry point -> owning precision zone
_DTN_ENTRY_ZONES = {"value": "jastrow_eval", "grad_lap": "jastrow_grad_lap", "ratio": "jastrow_ratio"}
# Primitives whose output dtype is the arithmetic dtype of the kernel body.
_DTN_ARITHMETIC_PRIMITIVES = frozenset({"sqrt", "pow", "integer_pow"})


def _dtn_entry_inputs(geometry):
    """Seeded electrons clustered around nucleus 0, so every channel and the moved electron interact.

    An electron outside every cutoff contributes exact zeros (and a ratio of exactly 1), which an
    fp64 kernel would also return; the cluster keeps every compared value generic.
    """
    data = _cusp_probe_dtn(geometry, 4, "L2.5-high-order")
    nucleus = np.asarray(data.structure_data.positions, dtype=np.float64)[0]
    rng = np.random.default_rng(110)
    r_up = nucleus + rng.normal(scale=0.4, size=(3, 3))
    r_dn = nucleus + rng.normal(scale=0.4, size=(2, 3))
    new = r_up[0] + np.array([0.3, -0.2, 0.1])
    return data, r_up, r_dn, new


def _xla_dtn_entry(entry, data):
    if entry == "value":
        return lambda ru, rd, new: compute_Jastrow_DTN(data, ru, rd)
    if entry == "grad_lap":
        return lambda ru, rd, new: compute_grads_and_laplacian_Jastrow_DTN(data, ru, rd)

    jd = Jastrow_data(jastrow_dtn_data=data)

    def ratio(ru, rd, new):
        new_up = ru.at[0].set(new)
        state = _init_jastrow_dtn_ratio_state(data, ru, rd)
        return _compute_ratio_Jastrow_part_rank1_update(jd, ru, rd, new_up[None], rd[None], dtn_state=state)

    return ratio


def _rust_dtn_entry(entry, data, r_up, r_dn, new):
    if entry == "value":
        return np.atleast_1d(np.asarray(jqmc_rs.compute_dtn_value(data, r_up, r_dn), dtype=np.float64))
    if entry == "grad_lap":
        outs = jqmc_rs.compute_grads_and_laplacian_dtn(data, r_up, r_dn)
        return np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in outs])
    return np.atleast_1d(np.asarray(jqmc_rs.compute_dtn_ratio(data, r_up, r_dn, True, 0, new.tolist()), dtype=np.float64))


def _jaxpr_float_dtypes(closed_jaxpr, primitives):
    """Output dtypes of ``primitives`` anywhere in a (nested) jaxpr."""
    found = set()

    def as_jaxpr(obj):
        if hasattr(obj, "eqns"):
            return obj
        inner = getattr(obj, "jaxpr", None)
        return inner if hasattr(inner, "eqns") else None

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            for param in eqn.params.values():
                for sub in param if isinstance(param, (list, tuple)) else (param,):
                    inner = as_jaxpr(sub)
                    if inner is not None:
                        walk(inner)
            if eqn.primitive.name in primitives:
                found.update(str(v.aval.dtype) for v in eqn.outvars)

    walk(closed_jaxpr.jaxpr)
    return found


def _rust_arithmetic_dtype(values):
    """Tolerance-free witness of the dtype a Rust kernel computed in.

    An fp32 kernel widens its fp32 result to the f64 ABI, so every element is
    exactly representable in fp32.  An fp64 kernel's results are not, except
    by coincidence on a measure-zero set (the seeded inputs avoid it).
    """
    exact_f32 = np.all(values.astype(np.float32).astype(np.float64) == values)
    return "float32" if exact_f32 else "float64"


class TestDTNZoneDtypeConformance:
    """Each DTN kernel entry point computes in its zone's declared dtype on BOTH backends.

    A standalone ``jqmc_rs`` DTN kernel reads the precision captured at
    construction (``configure_native_precision``), not ``jqmc._precision``;
    a harness that configured only Python compared an fp64 Rust kernel with an
    fp32 XLA kernel.  Backend parity cannot see that (the two agree to fp32
    accuracy), so the dtype is asserted directly: XLA by the dtype of the
    kernel's arithmetic primitives and of its result, Rust by the exact fp32
    representability of its result.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("entry", sorted(_DTN_ENTRY_ZONES))
    @pytest.mark.parametrize("geometry", ["molecular", "periodic"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
    def test_dtn_entry_computes_in_declared_zone_dtype(self, geometry, entry):
        """XLA arithmetic, XLA result and Rust arithmetic all use ``get_dtype_np(zone)``."""
        declared = str(np.dtype(get_dtype_np(_DTN_ENTRY_ZONES[entry])))
        data, r_up, r_dn, new = _dtn_entry_inputs(geometry)
        fn = _xla_dtn_entry(entry, data)
        args = (jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(new))

        xla_arithmetic = _jaxpr_float_dtypes(jax.make_jaxpr(fn)(*args), _DTN_ARITHMETIC_PRIMITIVES)
        assert xla_arithmetic == {declared}, (geometry, entry, "xla-arithmetic", xla_arithmetic, declared)
        xla_results = {str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(fn(*args))}
        assert xla_results == {declared}, (geometry, entry, "xla-result", xla_results, declared)
        rust_arithmetic = _rust_arithmetic_dtype(_rust_dtn_entry(entry, data, r_up, r_dn, new))
        assert rust_arithmetic == declared, (geometry, entry, "rust-arithmetic", rust_arithmetic, declared)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("entry", ["value", "grad_lap"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
    def test_dtype_verdict_rejects_rust_forced_to_the_other_dtype(self, entry):
        """Negative control: the Rust kernel built under the opposite precision policy fails the verdict."""
        self.test_dtn_entry_computes_in_declared_zone_dtype("molecular", entry)
        python_policy = precision_policy_snapshot()
        flipped = dict(python_policy)
        zone = _DTN_ENTRY_ZONES[entry]
        flipped[zone] = "float64" if python_policy[zone] == "float32" else "float32"
        try:
            jqmc_rs.configure_native_precision(flipped)
            with pytest.raises(AssertionError, match="rust-arithmetic"):
                self.test_dtn_entry_computes_in_declared_zone_dtype("molecular", entry)
        finally:
            jqmc_rs.configure_native_precision(python_policy)

    @pytest.mark.light
    @pytest.mark.parametrize("entry", ["value", "grad_lap"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
    def test_dtype_verdict_rejects_xla_forced_to_the_other_dtype(self, monkeypatch, entry):
        """Negative control: the XLA kernel traced with the opposite zone dtype fails the verdict."""
        self.test_dtn_entry_computes_in_declared_zone_dtype("molecular", entry)
        zone = _DTN_ENTRY_ZONES[entry]
        other = jnp.float64 if np.dtype(get_dtype_np(zone)) == np.float32 else jnp.float32
        real_get_dtype_jnp = jastrow_factor_module.get_dtype_jnp
        monkeypatch.setattr(
            jastrow_factor_module,
            "get_dtype_jnp",
            lambda name: other if name == zone else real_get_dtype_jnp(name),
        )
        jax.clear_caches()
        try:
            with pytest.raises(AssertionError, match="xla-arithmetic"):
                self.test_dtn_entry_computes_in_declared_zone_dtype("molecular", entry)
        finally:
            jax.clear_caches()


# ---------------------------------------------------------------------------
# 5. J1 + J2 on the triclinic cell: value + ratio parity (HOLE 2)
# ---------------------------------------------------------------------------


class TestJ1J2TriclinicParity:
    @staticmethod
    def _build_j1_j2(sd, j1b_type="exp", j2b_type="pade"):
        n_atoms = len(np.asarray(sd.positions))
        j1 = Jastrow_one_body_data(
            jastrow_1b_param=0.8,
            jastrow_1b_type=j1b_type,
            structure_data=sd,
            # Zero is the CORRECT value here, not the hardcoded placeholder that
            # tests/_ecp_core_electrons.py exists to replace: this structure is built
            # in-test with no Coulomb_potential_data at all, so there is no
            # pseudopotential and no core to remove. There is nothing to read it from.
            core_electrons=tuple([0.0] * n_atoms),
        )
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.9, jastrow_2b_type=j2b_type, structure_data=sd)
        return j1, j2

    @pytest.mark.light
    @pytest.mark.parametrize("is_up", [True, False])
    @pytest.mark.parametrize("seed", [1, 7])
    def test_j1_j2_triclinic_parity(self, is_up, seed):
        """Rust ``compute_Jastrow_one_body`` / ``compute_Jastrow_two_body``
        values and the ``jastrow_1b2b_ratio`` single-move ratio match XLA on
        the battery's triclinic cell, on in-cell AND boundary-crossing
        configurations."""
        av, rv = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
        ar, rr = get_tolerance_xla_rust_consistency(_ZONES_RATIO, level="strict")
        sd = _JastrowPBCAnalytic._pbc_structure(triclinic=True)
        j1, j2 = self._build_j1_j2(sd)
        jd = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2)

        cell = _cell_of(sd)
        rng = np.random.default_rng(seed)
        n_up, n_dn = 5, 4
        r_up = (rng.random((n_up, 3)) @ cell).astype(np.float64)
        r_dn = (rng.random((n_dn, 3)) @ cell).astype(np.float64)

        # In-cell value parity.
        j1_x = float(compute_Jastrow_one_body(j1, r_up, r_dn))
        j1_r = float(jqmc_rs.compute_Jastrow_one_body(j1, r_up, r_dn))
        j2_x = float(compute_Jastrow_two_body(j2, r_up, r_dn))
        j2_r = float(jqmc_rs.compute_Jastrow_two_body(j2, r_up, r_dn))
        assert np.isclose(j1_r, j1_x, atol=av, rtol=rv), ("j1", j1_r, j1_x)
        assert np.isclose(j2_r, j2_x, atol=av, rtol=rv), ("j2", j2_r, j2_x)

        # Boundary-crossing value parity + lattice invariance (J2 uses the
        # periodic distance; J1 the nearest-atom min image -- both wrap).
        dr_up, dr_dn = _displace_by_lattice(sd, r_up, r_dn, seed)
        j1_xd = float(compute_Jastrow_one_body(j1, dr_up, dr_dn))
        j1_rd = float(jqmc_rs.compute_Jastrow_one_body(j1, dr_up, dr_dn))
        j2_xd = float(compute_Jastrow_two_body(j2, dr_up, dr_dn))
        j2_rd = float(jqmc_rs.compute_Jastrow_two_body(j2, dr_up, dr_dn))
        assert np.isclose(j1_rd, j1_xd, atol=av, rtol=rv), ("j1-disp", j1_rd, j1_xd)
        assert np.isclose(j2_rd, j2_xd, atol=av, rtol=rv), ("j2-disp", j2_rd, j2_xd)
        assert np.isclose(j2_xd, j2_x, atol=av, rtol=rv), ("j2-invariance", j2_xd, j2_x)
        assert np.isclose(j2_rd, j2_r, atol=av, rtol=rv), ("j2-rust-invariance", j2_rd, j2_r)

        # Single-move ratio parity across the cell boundary.
        idx = 0
        base = r_up[idx] if is_up else r_dn[idx]
        new = (base + 0.4 * cell[1] + rng.normal(scale=0.3, size=3)).astype(np.float64)
        new_up = r_up.copy()
        new_dn = r_dn.copy()
        if is_up:
            new_up[idx] = new
        else:
            new_dn[idx] = new
        truth = float(np.exp(compute_Jastrow_part(jd, new_up, new_dn) - compute_Jastrow_part(jd, r_up, r_dn)))
        rank1 = float(
            np.asarray(_compute_ratio_Jastrow_part_rank1_update(jd, r_up, r_dn, new_up[None, ...], new_dn[None, ...]))[0]
        )
        rust = float(jqmc_rs.jastrow_1b2b_ratio(jd, r_up, r_dn, is_up, idx, new))
        assert np.isfinite(rust)
        # Backend parity compares the same rank-1 route.  In fp32, a full
        # J(new)-J(old) recomputation has a deliberately different reduction
        # tree, so its cancellation is a normal numerical-consistency check,
        # not an XLA=Rust backend-consistency check.
        assert np.isclose(rust, rank1, atol=ar, rtol=rr), ("ratio-parity", rust, rank1)
        an, rn = get_tolerance_min(_ZONES_RATIO, level="strict")
        assert np.isclose(rust, truth, atol=an, rtol=rn), ("ratio-recompute", rust, truth)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Issue #68: Rust DTN on oblique lattices with the true WS-radius defaults.
# ---------------------------------------------------------------------------


@pytest.mark.light
def test_rust_periodic_dtn_default_value_is_basis_independent_on_oblique_lattices():
    """Rust DTN value, grads and Laplacians are basis independent and match XLA on oblique lattices.

    Uses the fixture of the XLA test
    ``test_periodic_dtn_default_cutoffs_and_value_are_basis_independent_on_oblique_lattices``
    (default cutoffs = 0.95 x the WS inscribed radius, electrons placed where the plain
    parallelepiped wrap picks a longer image inside a cutoff; three bases per lattice, one
    of them far from reduced). For each lattice, the Rust kernels in every basis must
    equal the XLA production values in the first basis. Grad/lap backend parity
    uses the independent-fp32 pair for two separately rounded implementations;
    Rust basis invariance keeps the strict pair. The XLA reference is itself
    checked against brute-force minimum-image distances and for basis
    independence by the XLA test.
    """
    from tests.test_jqmc_kernel.test_jastrow_dtn import DTN_WS_LATTICE_CASES, _ws_basis_fixture

    atol_v, rtol_v = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
    atol_g, rtol_g = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    pair_atol_g, pair_rtol_g = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="independent_fp32")
    for lattice_name, basis_names in DTN_WS_LATTICE_CASES:
        reference = None
        rust_reference = None
        for basis_name in basis_names:
            *_, dtn, r_up, r_dn = _ws_basis_fixture(lattice_name, basis_name)
            label = f"{lattice_name}/{basis_name}"
            if reference is None:
                reference = (
                    float(compute_Jastrow_DTN(dtn, jnp.asarray(r_up), jnp.asarray(r_dn))),
                    [
                        np.asarray(x, dtype=np.float64)
                        for x in compute_grads_and_laplacian_Jastrow_DTN(dtn, jnp.asarray(r_up), jnp.asarray(r_dn))
                    ],
                )
            value = float(jqmc_rs.compute_dtn_value(dtn, r_up, r_dn))
            grads_laps = [np.asarray(x, dtype=np.float64) for x in jqmc_rs.compute_grads_and_laplacian_dtn(dtn, r_up, r_dn)]
            if rust_reference is None:
                rust_reference = grads_laps
            np.testing.assert_allclose(value, reference[0], atol=atol_v, rtol=rtol_v, err_msg=f"{label}: J")
            for name, got, ref, basis_ref in zip(
                ("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_laps, reference[1], rust_reference
            ):
                np.testing.assert_allclose(got, ref, atol=pair_atol_g, rtol=pair_rtol_g, err_msg=f"{label}: {name}")
                np.testing.assert_allclose(got, basis_ref, atol=atol_g, rtol=rtol_g, err_msg=f"{label}: Rust basis {name}")
        jax.clear_caches()

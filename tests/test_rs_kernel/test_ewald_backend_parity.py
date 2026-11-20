"""Backend parity for the Ewald lattice-Coulomb channels: production JAX (Python
``jqmc.coulomb_potential``) vs native Rust (``jqmc_rs``), on BULK and SLAB/VACUUM
cells.

Why this file exists
--------------------
An Ewald sum that is not converged on slab/vacuum cells is a defect of the Python
table builder that the Rust kernels consume. Elsewhere:

* ``tests/test_rs_kernel/test_ewald.py`` exercises the Rust ee/ei kernels ONLY on
  compact bulk Gamma fixtures (``real_image_list`` has a handful of images), and
  does **not** cover the ii channel at all.
* the physics-level Ewald battery in
  ``tests/test_jqmc_kernel/test_coulomb_potential.py`` (pymatgen ii references,
  eta-invariance of the total, ee lattice-translation invariance, the
  ``TestEwaldSlabVacuum`` acceptance set) runs ONLY through the JAX path.

So a change that touches only the Python table builder could leave the Rust
sampling path -- the PRODUCTION engine on CPU, ``nw=128`` per rank -- silently
wrong. The slab tables (~400-image ``real_image_list``, 1-2 orders fewer
G-vectors) flow into the SAME Rust kernels; this file proves they are consumed
correctly on the slab/vacuum cells and fails if the Rust side diverges.

Design
------
* Every physics quantity (ii, ee, ei; total eta-invariance; ee translation
  invariance; ii-vs-pymatgen) is evaluated through BOTH backends via
  :func:`ewald_ii` / :func:`ewald_ee` / :func:`ewald_ei`, parametrized over
  ``BACKENDS = ("jax", "rust")``.
* Fixtures and reference constants are IMPORTED from
  ``tests.test_jqmc_kernel.test_coulomb_potential`` -- no reference table is
  duplicated here.
* Red-not-skip: when Rust is the resolved production backend (the driver's own
  ``flag_rust_on_CPU AND cpu AND full-precision`` rule) but ``jqmc_rs`` is
  missing or STALE, the ``rust`` parametrization FAILS with a build instruction.
  Only an explicit opt-out (``JQMC_FLAG_RUST_ON_CPU=0`` / GPU / mixed precision)
  turns it into a documented skip.
* Staleness ("installed but built from older sources") reuses the crate's
  build-identity digest ``jqmc_rs.__source_fnv64__`` (baked by ``jqmc/rs/build.rs``,
  recomputed by ``tests/test_rs_kernel/test_build_identity.py``).
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

import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jqmc._setting import ATOL_EWALD_BACKEND_PARITY, RTOL_EWALD_BACKEND_PARITY, flag_rust_on_CPU
from jqmc.coulomb_potential import (
    Coulomb_potential_data,
    build_ewald_data,
    compute_ewald_potential_ee,
    compute_ewald_potential_ei,
    compute_ewald_potential_ii,
)

# Fixture builders and pymatgen ii references are the single source of truth in
# the physics test module; import them rather than re-encode any reference.
from tests.test_jqmc_kernel.test_coulomb_potential import (
    PYMATGEN_E_II_NACL_SC_HA,
    PYMATGEN_E_II_NACL_SLAB_VAC_HA,
    PYMATGEN_E_II_WATER_LIH_ADS_HA,
    PYMATGEN_E_II_WATER_LIH_FAR_HA,
    _make_h2_box_fixture,
    _make_h4_slab_box_fixture,
    _make_nacl_sc_fixture,
    _make_nacl_slab_structure,
    _make_water_lih_structure,
)

# ``without_chdir``: every fixture builds in-memory from imported helpers and no
# path is relative, so this module manages its own CWD (opting out of the autouse
# ``_isolate_cwd`` chdir), like ``tests/test_rs_kernel/test_build_identity.py``.
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# --------------------------------------------------------------------------- #
# Tolerance constants (documented; house style = a named constant with a
# rationale, cf. jqmc/_setting.py::ATOL_EWALD_VS_EXTERNAL_REFERENCE et al.)
# --------------------------------------------------------------------------- #

#: Absolute tolerance for a direct JAX-vs-Rust channel comparison (ii / ee / ei)
#: on IDENTICAL Ewald tables. The two paths run the same erfc image sum +
#: reciprocal structure factor in fp64; the only difference is summation order
#: (JAX fused reductions vs Rust scalar loops), so the residual is pure fp
#: reassociation. Measured max |diff| across all fixtures here (bulk NaCl, NaCl
#: slab, 19-atom slab far/ads, H4 slab) is ~5e-14 abs (ii/ei) and ~6e-15 (ee).
#: 1e-10 gives >1000x headroom while still catching a real port regression: the
#: slab truncation bug this file guards against was ~1e-3 RELATIVE on ii
#: (tens of mHa on the channels), i.e. 7-8 orders above this floor.
EWALD_BACKEND_PARITY_ATOL = ATOL_EWALD_BACKEND_PARITY
#: Companion relative tolerance, applied to ``max(1, |reference|)`` so the gate
#: does not tighten below the absolute floor for the O(10-60 Ha) charged-slab ii.
EWALD_BACKEND_PARITY_RTOL = RTOL_EWALD_BACKEND_PARITY

#: eta-invariance of the TOTAL Ewald energy (E_ii + E_ee + E_ei) is an exact
#: identity; the per-channel truncation is pinned to the builder ``tol`` (1e-10
#: here), so the total agrees across eta to a few x that. Matches the strict
#: intent of the JAX-only analogue
#: (``test_compute_ewald_potential_total_energy_eta_invariance_*``), loosened by
#: one decade to absorb the extra fp reassociation of the Rust scalar loops.
EWALD_ETA_INVARIANCE_ATOL = 1.0e-7

#: ee lattice-translation invariance (shift ALL electrons by a lattice vector):
#: exact by Bloch periodicity. Same 1e-9/1e-9 the JAX analogue uses
#: (jqmc/_setting.py::{A,R}TOL_PBC_TRANSLATION_INVARIANT), which this mirrors.
EWALD_TRANSLATION_ATOL = 1.0e-9
EWALD_TRANSLATION_RTOL = 1.0e-9

BACKENDS = ("jax", "rust")

# --------------------------------------------------------------------------- #
# Rust availability / staleness gate (a failure, not a skip, when Rust is expected)
# --------------------------------------------------------------------------- #


def _rust_parity_is_required() -> bool:
    """Whether this CPU build promises the Rust Ewald implementation.

    Mixed precision selects XLA for the high-level driver, but it does not make
    the compiled Rust Ewald implementation disappear.  Backend parity is a
    capability contract, so both precision lanes execute it; mixed must not
    turn the comparison into a pytest skip.
    """
    env = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if env is not None:
        flag = env.strip().lower() not in ("0", "false", "no", "off", "")
    else:
        flag = flag_rust_on_CPU
    return flag and jax.default_backend() == "cpu"


_RUST_EXPECTED = _rust_parity_is_required()

try:
    import jqmc_rs  # noqa: E402

    _RUST_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only on a broken build
    jqmc_rs = None  # type: ignore[assignment]
    _RUST_IMPORT_ERROR = exc


def _rust_source_digest_mismatch() -> str | None:
    """Return a message if the loaded ``jqmc_rs`` was built from DIFFERENT crate
    sources than the working tree, else ``None``.

    Reuses the canonical FNV-1a digest helper from the build-identity guard
    (``tests/test_rs_kernel/test_build_identity.py``, itself lock-stepped with
    ``jqmc/rs/build.rs``) so there is exactly one digest scheme. The import is
    lazy and only happens once Rust is confirmed importable, so it never defeats
    the red-not-skip contract (that module imports ``jqmc_rs`` unconditionally).
    """
    embedded = getattr(jqmc_rs, "__source_fnv64__", None)
    if embedded is None:
        return (
            "jqmc_rs has no __source_fnv64__ attribute -- the imported extension predates the "
            "build-identity guard (jqmc/rs/build.rs). Rebuild: pip install -e . --no-build-isolation."
        )
    try:
        from tests.test_rs_kernel.test_build_identity import _source_fnv64
    except Exception:  # pragma: no cover - crate tree absent (not our concern here)
        return None
    recomputed = _source_fnv64()
    if embedded != recomputed:
        return (
            f"STALE jqmc_rs: embedded source digest {embedded} != current crate digest "
            f"{recomputed}. The loaded extension was built from a DIFFERENT revision of jqmc/rs "
            "than the working tree, so this parity result is against the wrong binary. Rebuild: "
            "pip install -e . --no-build-isolation."
        )
    return None


def _require_rust() -> None:
    """Gate a Rust parity parametrization, independently of precision mode."""
    if not _RUST_EXPECTED:
        pytest.skip("Rust parity is explicitly opted out (JQMC_FLAG_RUST_ON_CPU off or GPU backend).")
    if jqmc_rs is None:
        pytest.fail(
            "Rust backend expected (flag_rust_on_CPU + CPU + full precision) but 'jqmc_rs' is "
            f"not importable -- build jqmc/rs (pip install -e . --no-build-isolation). Import "
            f"error: {_RUST_IMPORT_ERROR!r}"
        )
    stale = _rust_source_digest_mismatch()
    if stale is not None:
        pytest.fail(stale)


# --------------------------------------------------------------------------- #
# Fixtures: (structure_data, coulomb_potential_data, r_up, r_dn), then ewald.
# All point-charge; electrons are synthetic (parity does not need a wavefunction).
# --------------------------------------------------------------------------- #


def _electrons(structure, n_up=6, n_dn=6, seed=7):
    cell = np.asarray(structure.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = np.asarray(rng.random((n_up, 3)) @ cell, dtype=np.float64)
    r_dn = np.asarray(rng.random((n_dn, 3)) @ cell, dtype=np.float64)
    return r_up, r_dn


def _fixture(name):
    """Return (structure, coulomb_base, r_up, r_dn) for a named cell."""
    if name == "nacl_sc_bulk":
        s, c = _make_nacl_sc_fixture()
        r_up, r_dn = _electrons(s)
    elif name == "h2_box_bulk":
        s, c, r_up, r_dn = _make_h2_box_fixture()
    elif name == "nacl_slab":
        s = _make_nacl_slab_structure()
        c = Coulomb_potential_data(structure_data=s, ecp_flag=False)
        r_up, r_dn = _electrons(s)
    elif name == "water_lih_far":
        s = _make_water_lih_structure("far")
        c = Coulomb_potential_data(structure_data=s, ecp_flag=False)
        r_up, r_dn = _electrons(s)
    elif name == "water_lih_ads":
        s = _make_water_lih_structure("ads")
        c = Coulomb_potential_data(structure_data=s, ecp_flag=False)
        r_up, r_dn = _electrons(s)
    elif name == "h4_slab":
        s, c, r_up, r_dn = _make_h4_slab_box_fixture()
    else:  # pragma: no cover
        raise KeyError(name)
    return s, c, r_up, r_dn


def _built(name, tol=1e-10, eta_override=None):
    s, c, r_up, r_dn = _fixture(name)
    ed = build_ewald_data(s, c, tol=tol, eta_override=eta_override)
    cp = c.replace(ewald_data=ed)
    return s, cp, ed, r_up, r_dn


#: All Ewald fixtures: one compact bulk cubic + one bulk box + the three slab/
#: vacuum cells that the slab Ewald fix targeted.
ALL_FIXTURES = [
    "nacl_sc_bulk",
    "h2_box_bulk",
    "nacl_slab",
    "water_lih_far",
    "water_lih_ads",
    "h4_slab",
]
SLAB_FIXTURES = ["nacl_slab", "water_lih_far", "water_lih_ads", "h4_slab"]

# --------------------------------------------------------------------------- #
# Backend dispatch (the parametrization mechanism).
# --------------------------------------------------------------------------- #


def _tables(structure, ed):
    return (
        float(ed.eta),
        np.asarray(ed.real_image_list, dtype=np.float64),
        np.asarray(ed.g_vectors, dtype=np.float64),
        np.asarray(ed.g_factors, dtype=np.float64),
        np.asarray(structure.cell, dtype=np.float64),
        int(structure._n_max),
    )


def ewald_ii(backend, cp, ed):
    """Ion-ion Ewald energy through ``backend``."""
    if backend == "jax":
        return float(compute_ewald_potential_ii(cp, ed))
    _require_rust()
    s = cp.structure_data
    eta, ri, gv, gf, cell, _ = _tables(s, ed)
    charges = np.asarray(cp._effective_charges, dtype=np.float64)
    positions = np.asarray(s._positions_cart_np, dtype=np.float64)
    return float(jqmc_rs.compute_ewald_potential_ii(charges, positions, eta, ri, gv, gf, cell))


def ewald_ee(backend, r_up, r_dn, cp, ed):
    """Electron-electron Ewald energy through ``backend``."""
    if backend == "jax":
        return float(compute_ewald_potential_ee(r_up, r_dn, ed))
    _require_rust()
    eta, ri, gv, gf, cell, nmax = _tables(cp.structure_data, ed)
    return float(jqmc_rs.compute_ewald_potential_ee(r_up, r_dn, eta, ri, gv, gf, cell, nmax))


def ewald_ei(backend, r_up, r_dn, cp, ed):
    """Electron-ion Ewald energy (summed over electrons) through ``backend``."""
    if backend == "jax":
        up, dn = compute_ewald_potential_ei(cp, r_up, r_dn, ed)
        return float(np.sum(np.asarray(up)) + np.sum(np.asarray(dn)))
    _require_rust()
    s = cp.structure_data
    eta, ri, gv, gf, cell, nmax = _tables(s, ed)
    z = np.asarray(cp._effective_charges, dtype=np.float64)
    r_ion = np.asarray(s._positions_cart_np, dtype=np.float64)
    return float(jqmc_rs.compute_ewald_potential_ei(r_up, r_dn, eta, ri, gv, gf, cell, nmax, z, r_ion))


_CHANNEL = {"ii": None, "ee": None, "ei": None}


def _channel_value(channel, backend, cp, ed, r_up, r_dn):
    if channel == "ii":
        return ewald_ii(backend, cp, ed)
    if channel == "ee":
        return ewald_ee(backend, r_up, r_dn, cp, ed)
    return ewald_ei(backend, r_up, r_dn, cp, ed)


# --------------------------------------------------------------------------- #
# 1. Direct JAX-vs-Rust parity, every channel, every cell.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
@pytest.mark.parametrize("channel", ["ii", "ee", "ei"])
def test_ewald_channel_backend_parity(fixture, channel):
    """ii/ee/ei agree between the JAX and Rust paths on identical Ewald tables,
    on bulk AND slab/vacuum cells (the regime of the slab Ewald fix)."""
    _s, cp, ed, r_up, r_dn = _built(fixture)
    val_jax = _channel_value(channel, "jax", cp, ed, r_up, r_dn)
    val_rust = _channel_value(channel, "rust", cp, ed, r_up, r_dn)
    assert np.isfinite(val_jax) and np.isfinite(val_rust)
    tol = EWALD_BACKEND_PARITY_ATOL + EWALD_BACKEND_PARITY_RTOL * max(1.0, abs(val_jax))
    assert abs(val_jax - val_rust) <= tol, (
        f"{channel} JAX={val_jax!r} vs Rust={val_rust!r} on {fixture}: |diff|={abs(val_jax - val_rust):.3e} > {tol:.3e}"
    )


# --------------------------------------------------------------------------- #
# 2. Backend availability gate (red, not skip).
#    NOTE: physics-level correctness (external
#    references, invariance identities) is validated on the XLA path only;
#    the Rust backend's contract is exactly Rust == XLA, which is what the
#    channel parity tests above enforce on regime-diverse fixtures
#    (bulk + slab/vacuum).
# --------------------------------------------------------------------------- #


def test_rust_backend_gate_is_red_not_skip_when_expected():
    """A promised Rust parity backend is importable and source-fresh in both
    full and mixed test lanes; precision selection cannot hide a stale build."""
    if not _RUST_EXPECTED:
        pytest.skip("Rust parity explicitly opted out (JQMC_FLAG_RUST_ON_CPU off / GPU).")
    assert jqmc_rs is not None, (
        "Rust backend expected but 'jqmc_rs' not importable -- build jqmc/rs "
        f"(pip install -e . --no-build-isolation). Import error: {_RUST_IMPORT_ERROR!r}"
    )
    assert hasattr(jqmc_rs, "compute_ewald_potential_ii"), (
        "jqmc_rs lacks compute_ewald_potential_ii -- the loaded extension predates the "
        "ii parity binding; rebuild the crate from current source."
    )
    stale = _rust_source_digest_mismatch()
    assert stale is None, stale

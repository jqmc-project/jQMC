"""Backend parity for the Ewald FORCE and STRESS channels (dE/dr, dE/dR, dE/deps):
production JAX analytic derivatives (``jqmc.derivatives.ewald``) vs native Rust
(``jqmc_rs``), on BULK, SLAB/VACUUM and MONOCLINIC cells.

Why this file exists
--------------------
The derivative twin of ``test_ewald_backend_parity.py`` (the energy-channel
template), which pins the ENERGY channels' Rust==XLA parity on slab/vacuum cells.
The other Rust force/stress parity tests use compact bulk cells only:

* ``tests/test_rs_kernel/test_forces_ewald.py`` runs ``grad_Eee_wrt_r`` /
  ``grad_Eei`` / ``grad_Eii_wrt_R`` and the three ``_wrt_strain`` kernels ONLY on
  ``SYS = ["pbc_composite_jnone", "twist_sphepgto_jnone"]`` -- two bulk lithium
  cells;
* the physics battery validates Ewald ii/ei/ee references on slab/vacuum cells
  (``TestEwaldSlabVacuum``) and on non-orthorhombic cells (CuO tenorite, ZnO
  wurtzite, monoclinic H2), but only through the JAX path.

So a slab-regime divergence in the Rust force/stress kernels -- an unconverged
slab table, or a min-image folding error on a non-orthogonal / high-aspect cell
(raw-vs-min-image ``r_i - R_I`` in the ei force and ei strain, invisible while
walkers stay home) -- is caught by no other test. These channels feed the production VMC/GFMC
atomic forces and the stress tensor.

Design (same contract as the template)
--------------------------------------
* The XLA reference is the PRODUCTION analytic-derivative module
  ``jqmc.derivatives.ewald`` (what the jitted force/stress observables actually
  evaluate), not ``jax.grad`` of the energy: the contract is Rust == XLA.
* Fixtures are imported from the energy template / the physics module -- the two
  bulk cells, all four slab/vacuum cells, plus the monoclinic H2 cell (the only
  committed non-orthorhombic Ewald fixture; it stands in for the CuO/ZnO
  anisotropy regime of the ii reference battery).
* Each quantity runs on a NEAR (in-cell) and a FAR-ROAMED electron configuration
  (every electron displaced by a random integer lattice combination, vacuum axis
  included) so the raw-vs-min-image bug class is live on the anisotropic cells.
* Red-not-skip + staleness: reuses ``_require_rust`` from the template.
* Tolerances come from the precision-zone registry (``get_tolerance_min``), the
  same "ewald" zone and levels the bulk parity file uses -- no bare constants.
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

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_RELATIVE_COMPARISON_SCALE_FLOOR
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
from jqmc.derivatives import ewald as jqmc_ewald

# Fixture builders, backend gate and table extraction are shared with the energy
# template; nothing is re-encoded here.
from tests.test_jqmc_kernel.test_coulomb_potential import _make_h2_monoclinic_fixture
from tests.test_rs_kernel.test_ewald_backend_parity import (
    ALL_FIXTURES,
    _built,
    _require_rust,
    _tables,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

#: Precision zone shared with tests/test_rs_kernel/test_forces_ewald.py: the
#: whole comparison lives in the Ewald lattice-sum zone.
_ZONE = ("ewald",)

#: Tolerance LEVELS mirror the bulk parity file (test_forces_ewald.py): "medium"
#: for the force channels and the far-roamed strain, "strict" for the in-cell
#: strain (ion/lattice-only path, no AO-cutoff truncation). The slab bug class
#: this file guards against was ~1e-3 RELATIVE on the energy channels and O(1-10)
#: ABSOLUTE on far-roamed ei derivatives -- many orders above either level.

#: All Ewald cells: two bulk + four slab/vacuum (from the energy template) + the
#: monoclinic H2 cell (non-orthorhombic regime of the ii/ei/ee physics battery).
FIXTURES = ALL_FIXTURES + ["h2_monoclinic"]


def _built_any(name):
    """(structure, coulomb+ewald, ewald_data, r_up, r_dn) for every FIXTURES row."""
    if name == "h2_monoclinic":
        s, c, r_up, r_dn = _make_h2_monoclinic_fixture()
        ed = build_ewald_data(s, c, tol=1e-10)
        return s, c.replace(ewald_data=ed), ed, r_up, r_dn
    return _built(name)


def _roam(structure, r, seed):
    """Displace every electron by a random integer lattice combination (all three
    axes, so slab electrons roam into the vacuum images too)."""
    cell = np.asarray(structure.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return r + rng.integers(-2, 3, size=r.shape) @ cell


def _rust_args(cp, ed):
    """(eta, real_image_list, g_vectors, g_factors, cell, n_max, Z, R_ion) --
    the exact tuple the production driver feeds the Rust Ewald kernels."""
    s = cp.structure_data
    eta, ri, gv, gf, cell, nmax = _tables(s, ed)
    z = np.asarray(cp._effective_charges, dtype=np.float64)
    rion = np.asarray(s._positions_cart_np, dtype=np.float64)
    return eta, ri, gv, gf, cell, nmax, z, rion


def _assert_close(label, rust, ref, atol, rtol):
    rust, ref = np.asarray(rust), np.asarray(ref)
    d = np.abs(rust - ref).max()
    assert d <= atol or d <= rtol * (np.abs(ref).max() + ATOL_RELATIVE_COMPARISON_SCALE_FLOOR), (
        f"{label}: Rust vs XLA max|diff|={d:.3e} exceeds atol={atol:.1e} / rtol={rtol:.1e} (ref max|.|={np.abs(ref).max():.3e})"
    )


# --------------------------------------------------------------------------- #
# Forces: ee d/dr, ei d/d{r,R}, ii d/dR -- near and far-roamed electrons.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("roamed", [False, True], ids=["near", "far"])
def test_ewald_force_backend_parity(fixture, roamed):
    """dEee/dr, dEei/d{r,R}, dEii/dR agree between the production XLA analytic
    derivatives and the Rust kernels on bulk, slab/vacuum and monoclinic cells."""
    _require_rust()
    import jqmc_rs

    s, cp, ed, r_up, r_dn = _built_any(fixture)
    if roamed:
        r_up, r_dn = _roam(s, r_up, seed=11), _roam(s, r_dn, seed=13)
    eta, ri, gv, gf, cell, nmax, z, rion = _rust_args(cp, ed)
    atol, rtol = get_tolerance_min(_ZONE, "medium")

    _assert_close(
        f"ee force [{fixture}]",
        jqmc_rs.grad_Eee_wrt_r(r_up, r_dn, eta, ri, gv, gf, cell, nmax),
        jqmc_ewald.grad_Eee_wrt_r(ed, r_up, r_dn),
        atol,
        rtol,
    )
    dr, dR = jqmc_rs.grad_Eei(r_up, r_dn, eta, ri, gv, gf, z, rion, cell, nmax)
    _assert_close(f"ei d/dr [{fixture}]", dr, jqmc_ewald.grad_Eei_wrt_r(cp, ed, r_up, r_dn), atol, rtol)
    _assert_close(f"ei d/dR [{fixture}]", dR, jqmc_ewald.grad_Eei_wrt_R(cp, ed, r_up, r_dn), atol, rtol)
    _assert_close(
        f"ii d/dR [{fixture}]",
        jqmc_rs.grad_Eii_wrt_R(eta, ri, gv, gf, z, rion, cell),
        jqmc_ewald.grad_Eii_wrt_R(cp, ed),
        atol,
        rtol,
    )


# --------------------------------------------------------------------------- #
# Stress: ee/ei/ii d/deps -- near (strict) and far-roamed (medium) electrons.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("roamed", [False, True], ids=["near", "far"])
def test_ewald_stress_backend_parity(fixture, roamed):
    """dEee/deps, dEei/deps, dEii/deps agree between the production XLA analytic
    derivatives and the Rust kernels on bulk, slab/vacuum and monoclinic cells."""
    _require_rust()
    import jqmc_rs

    s, cp, ed, r_up, r_dn = _built_any(fixture)
    if roamed:
        r_up, r_dn = _roam(s, r_up, seed=17), _roam(s, r_dn, seed=19)
    eta, ri, gv, gf, cell, nmax, z, rion = _rust_args(cp, ed)
    atol, rtol = get_tolerance_min(_ZONE, "medium" if roamed else "strict")

    _assert_close(
        f"ee strain [{fixture}]",
        jqmc_rs.grad_Eee_wrt_strain(r_up, r_dn, eta, ri, gv, gf, cell, nmax),
        jqmc_ewald.grad_Eee_wrt_strain(ed, r_up, r_dn),
        atol,
        rtol,
    )
    _assert_close(
        f"ei strain [{fixture}]",
        jqmc_rs.grad_Eei_wrt_strain(r_up, r_dn, eta, ri, gv, gf, z, rion, cell, nmax),
        jqmc_ewald.grad_Eei_wrt_strain(cp, ed, r_up, r_dn),
        atol,
        rtol,
    )
    _assert_close(
        f"ii strain [{fixture}]",
        jqmc_rs.grad_Eii_wrt_strain(eta, ri, gv, gf, z, rion, cell),
        jqmc_ewald.grad_Eii_wrt_strain(cp, ed),
        atol,
        rtol,
    )

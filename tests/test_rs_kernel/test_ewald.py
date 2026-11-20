"""Layer-1 equivalence: Ewald lattice Coulomb (e-e, e-i), Rust vs jqmc.
Periodic V channels; tables (eta, real_image_list, g_vectors, g_factors) are
built in Python and consumed by Rust.

Fixtures are the committed hams via ``tests.test_rs_kernel._rs_fixtures.rs_ham``.
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

import jqmc_rs
import numpy as np
import pytest

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance
from tests.test_rs_kernel._rs_fixtures import rs_ham

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

# Small real (Gamma) periodic systems with an Ewald_data table: lithium (sphe
# pgto, ECP) and the QE Li pgto+pw composite.
SYS = ["pbc_gamma_sphepgto_jnone", "pbc_composite_jnone"]


def _tables(ham):
    cp = ham.coulomb_potential_data
    ed = cp.ewald_data
    cell = np.asarray(ham.structure_data.cell, float)
    return (
        cp,
        ed,
        float(ed.eta),
        np.asarray(ed.real_image_list, float),
        np.asarray(ed.g_vectors, float),
        np.asarray(ed.g_factors, float),
        cell,
        int(ham.structure_data._n_max),
    )


def _config(ham, seed=3):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = gem.num_electron_up, gem.num_electron_dn
    cell = np.asarray(ham.structure_data.cell, float)
    rng = np.random.default_rng(seed)
    r_up = np.asarray(rng.random((nu, 3)) @ cell, float)
    r_dn = np.asarray(rng.random((nd, 3)) @ cell, float)
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
def test_ewald_ee(pattern):
    """Ewald ee."""
    from jqmc.coulomb_potential import compute_ewald_potential_ee

    ham = rs_ham(pattern)
    _cp, ed, eta, ri, gv, gf, cell, nmax = _tables(ham)
    r_up, r_dn = _config(ham)
    e_rust = jqmc_rs.compute_ewald_potential_ee(r_up, r_dn, eta, ri, gv, gf, cell, nmax)
    e_jqmc = float(compute_ewald_potential_ee(r_up, r_dn, ed))
    atol, rtol = _d8_backend_tolerance(("ewald",), level="strict")
    assert abs(e_rust - e_jqmc) <= atol + rtol * max(1.0, abs(e_jqmc))


@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
def test_ewald_ei(pattern):
    """Ewald ei."""
    from jqmc.coulomb_potential import compute_ewald_potential_ei

    ham = rs_ham(pattern)
    cp, ed, eta, ri, gv, gf, cell, nmax = _tables(ham)
    z = np.asarray(cp._effective_charges, float)
    r_ion = np.asarray(ham.structure_data.positions, float)
    r_up, r_dn = _config(ham)
    e_rust = jqmc_rs.compute_ewald_potential_ei(r_up, r_dn, eta, ri, gv, gf, cell, nmax, z, r_ion)
    up, dn = compute_ewald_potential_ei(cp, r_up, r_dn, ed)
    e_jqmc = float(np.sum(up) + np.sum(dn))
    atol, rtol = _d8_backend_tolerance(("ewald",), level="strict")
    assert abs(e_rust - e_jqmc) <= atol + rtol * max(1.0, abs(e_jqmc))


# Full periodic e_L = T + V, where V = Ewald(ee+ei+ii) + ECP(local all-pairs +
# nonlocal NN=1). The periodic Jastrow (J1/J2) uses the Williamson 1997 smooth
# distance; the ECP min-image uses the Wigner-Seitz displacement.
#
# Reference = the EXACT slow path (full-LU det ratios, fp64). jqmc's production
# ``compute_local_energy`` instead routes the nonlocal ECP through
# ``..._nearest_neighbors_fast_update`` (Sherman-Morrison in the reduced
# "coulomb" zone), which for periodic+Jastrow deviates ~2e-6 from the exact
# value -- a fast-path approximation in jqmc, not a port discrepancy. The Rust
# port reproduces the exact reference to machine precision.
EL_SYS = [
    "pbc_gamma_sphepgto_jnone",
    "pbc_gamma_sphepgto_j12",
    "pbc_gamma_sphepgto_j123",
]


@pytest.mark.light
@pytest.mark.parametrize("pattern", EL_SYS)
def test_local_energy_periodic(pattern):
    """Local energy periodic."""
    from jqmc.coulomb_potential import (
        compute_bare_coulomb_potential,
        compute_ecp_local_parts_all_pairs,
        compute_ecp_non_local_parts_nearest_neighbors,
        octahedron_sym_mesh_Nv6,
    )
    from jqmc.wavefunction import compute_kinetic_energy

    ham = rs_ham(pattern)
    cp, wf = ham.coulomb_potential_data, ham.wavefunction_data
    r_up, r_dn = _config(ham)
    RT = np.eye(3)
    # Exact slow-path reference (full-LU, fp64).
    T = float(compute_kinetic_energy(wf, r_up, r_dn))
    bare = float(compute_bare_coulomb_potential(cp, r_up, r_dn))
    loc = float(compute_ecp_local_parts_all_pairs(cp, r_up, r_dn))
    *_, nl = compute_ecp_non_local_parts_nearest_neighbors(cp, wf, r_up, r_dn, RT, NN=1, Nv=6, flag_determinant_only=False)
    eL_exact = T + bare + loc + float(nl)

    w = np.array(octahedron_sym_mesh_Nv6.weights, float)
    gp = np.array(octahedron_sym_mesh_Nv6.grid_points, float) @ RT
    eL_rust = jqmc_rs.compute_local_energy(ham, r_up, r_dn, w, gp)
    # Matches the molecular test_local_energy tolerance; the slow-path
    # reference carries config-dependent fp noise from jqmc's reduced "coulomb"
    # zone mesh geometry, not a port offset. Full local energy crosses AO eval,
    # determinant eval, coulomb, ewald and kinetic zones -- take the loosest.
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_eval", "coulomb", "ewald", "wf_kinetic"), level="strict")
    assert abs(eL_rust - eL_exact) <= atol + rtol * max(1.0, abs(eL_exact))

"""Layer-1: periodic Ewald forces (Rust forces_ewald.rs) vs jax.grad(jqmc Ewald
potential). ee d/dr, ei d/d{r,R}, ii d/dR, + 3 stress tests. Real + complex-twist
cells (Ewald is real either way) -- Coulomb/lattice electrostatics does not depend
on the AO basis at all (only on the cell + ion positions + Ewald tables), so any
committed periodic fixture gives full coverage.

Fixtures are committed-fixture patterns via ``tests.test_rs_kernel._rs_fixtures.rs_ham``.
The stress oracle is ``jqmc.derivatives.ewald`` (grad_Eee_wrt_strain
/ grad_Eei_wrt_strain / grad_Eii_wrt_strain), the same
substitution pattern ``test_gfmc_complex_force_stress.py`` uses for
``jqmc.derivatives.gfmc``.
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

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc._setting import ATOL_RELATIVE_COMPARISON_SCALE_FLOOR  # noqa: E402
from jqmc.coulomb_potential import (  # noqa: E402
    compute_ewald_potential_ee,
    compute_ewald_potential_ei,
    compute_ewald_potential_ii,
)
from jqmc.derivatives import ewald as jqmc_ewald  # noqa: E402

import jqmc_rs  # noqa: E402

from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = pytest.mark.with_chdir

# Real-cell row (composite pgto+PW, jnone) + complex-twist row (k != Gamma sphe-pgto,
# jnone). Either real pattern works since Ewald ignores the AO family; the twist
# pattern is required to cover the complex-twist code path.
SYS = ["pbc_composite_jnone", "twist_sphepgto_jnone"]

_ZONE = ("ewald",)


def _tabs(ham):
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    return (
        cd,
        ed,
        float(ed.eta),
        np.asarray(ed.real_image_list, float),
        np.asarray(ed.g_vectors, float),
        np.asarray(ed.g_factors, float),
        np.asarray(ham.structure_data.cell, float),
        int(ham.structure_data._n_max),
        np.asarray(cd._effective_charges, float),
        np.asarray(ed.structure_data.positions, float),
    )


def _cfg(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = gem.num_electron_up, gem.num_electron_dn
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
    return ru, rd


def _close(a, b, atol, rtol):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a - b).max()
    return d <= atol or d <= rtol * (np.abs(b).max() + ATOL_RELATIVE_COMPARISON_SCALE_FLOOR)


@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
def test_ewald_ee_force(pattern):
    """Ewald ee force."""
    ham = rs_ham(pattern)
    cd, ed, eta, ri, gv, gf, cell, nmax, z, rion = _tabs(ham)
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_Eee_wrt_r(ru, rd, eta, ri, gv, gf, cell, nmax)
    gu = jax.grad(lambda r: compute_ewald_potential_ee(r, rd, ed))(ru)
    gd = jax.grad(lambda r: compute_ewald_potential_ee(ru, r, ed))(rd)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE)
    assert _close(ana, np.concatenate([np.asarray(gu), np.asarray(gd)], 0), atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
def test_ewald_ei_force(pattern):
    """Ewald ei force."""
    ham = rs_ham(pattern)
    cd, ed, eta, ri, gv, gf, cell, nmax, z, rion = _tabs(ham)
    ru, rd = _cfg(ham)
    dr, dR = jqmc_rs.grad_Eei(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE)

    def E(cd_, ewd_, a, b):
        eu, edn = compute_ewald_potential_ei(cd_, a, b, ewd_)
        return jnp.sum(eu) + jnp.sum(edn)

    gu = jax.grad(lambda r: E(cd, ed, r, rd))(ru)
    gd = jax.grad(lambda r: E(cd, ed, ru, r))(rd)
    assert _close(dr, np.concatenate([np.asarray(gu), np.asarray(gd)], 0), atol, rtol)
    pos = np.asarray(ed.structure_data.positions)

    def fwdR(P):
        cd2 = cd.replace(structure_data=cd.structure_data.replace(positions=P))
        ewd2 = ed.replace(structure_data=ed.structure_data.replace(positions=P))
        return E(cd2, ewd2, ru, rd)

    assert _close(dR, jax.grad(fwdR)(pos), atol, rtol)


# FAR-ROAMED e-i force: ``grad_Eei`` must MIN-IMAGE the electron-ion
# displacement ``r_i - R_I`` before the truncated ``T_n`` image sum, exactly as the ei
# ENERGY (``compute_ewald_potential_ei``). The near ``_cfg`` (scale 0.7, in-cell) never
# exposes it -- raw ``r_i - R_I - T_n`` == min-image while the walker stays home. Once an
# electron roams an integer lattice vector out, the raw sum misses the dominant near image
# and the ei force diverges O(10) from the energy's gradient. This row roams every electron
# by a random integer lattice combination so the raw-vs-min-image gap is live.
@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
@pytest.mark.parametrize("seed", [1, 3])
def test_ewald_ei_force_far(pattern, seed):
    """Ewald ei force far."""
    ham = rs_ham(pattern)
    cd, ed, eta, ri, gv, gf, cell, nmax, z, rion = _tabs(ham)
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    cellm = np.asarray(ham.structure_data.cell, float)  # rows = lattice vectors
    rng = np.random.default_rng(seed)

    def _roam(n):
        base = R[np.arange(n) % len(R)] + rng.normal(scale=0.4, size=(n, 3))
        return base + rng.integers(-2, 3, size=(n, 3)) @ cellm

    ru, rd = _roam(nu), _roam(nd)
    dr, dR = jqmc_rs.grad_Eei(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE)

    def E(cd_, ewd_, a, b):
        eu, edn = compute_ewald_potential_ei(cd_, a, b, ewd_)
        return jnp.sum(eu) + jnp.sum(edn)

    gu = jax.grad(lambda r: E(cd, ed, r, rd))(ru)
    gd = jax.grad(lambda r: E(cd, ed, ru, r))(rd)
    assert _close(dr, np.concatenate([np.asarray(gu), np.asarray(gd)], 0), atol, rtol)
    pos = np.asarray(ed.structure_data.positions)

    def fwdR(P):
        cd2 = cd.replace(structure_data=cd.structure_data.replace(positions=P))
        ewd2 = ed.replace(structure_data=ed.structure_data.replace(positions=P))
        return E(cd2, ewd2, ru, rd)

    assert _close(dR, jax.grad(fwdR)(pos), atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Ion-ion Ewald energy and force; no electron coordinates enter E_ii."
)
def test_ewald_ii_force(pattern):
    """Ewald ii force."""
    ham = rs_ham(pattern)
    cd, ed, eta, ri, gv, gf, cell, nmax, z, rion = _tabs(ham)
    ana = jqmc_rs.grad_Eii_wrt_R(eta, ri, gv, gf, z, rion, cell)
    pos = np.asarray(ed.structure_data.positions)
    ref = jax.grad(lambda P: compute_ewald_potential_ii(cd, ed.replace(structure_data=ed.structure_data.replace(positions=P))))(
        pos
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE)
    assert _close(ana, ref, atol, rtol)


# ---- Ewald stresses d E_{ee,ei,ii}/d eps (vs the productionized jqmc.derivatives.ewald
# analytic oracle). Signatures confirmed from jqmc/derivatives/ewald.py:
#   grad_Eee_wrt_strain(ewald_data, r_up, r_dn)
#   grad_Eei_wrt_strain(coulomb_data, ewald_data, r_up, r_dn)
#   grad_Eii_wrt_strain(coulomb_data, ewald_data)
@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
def test_ewald_stresses(pattern):
    """Ewald stresses."""
    ham = rs_ham(pattern)
    cd, ed, eta, ri, gv, gf, cell, nmax, z, rion = _tabs(ham)
    ru, rd = _cfg(ham)
    # Ion-position/lattice-only stresses (no AO-cutoff truncation on this path);
    # strict is the appropriate level here (it brackets a 1e-9 bare atol).
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE)
    assert _close(
        jqmc_rs.grad_Eee_wrt_strain(ru, rd, eta, ri, gv, gf, cell, nmax), jqmc_ewald.grad_Eee_wrt_strain(ed, ru, rd), atol, rtol
    )
    assert _close(
        jqmc_rs.grad_Eei_wrt_strain(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax),
        jqmc_ewald.grad_Eei_wrt_strain(cd, ed, ru, rd),
        atol,
        rtol,
    )
    assert _close(
        jqmc_rs.grad_Eii_wrt_strain(eta, ri, gv, gf, z, rion, cell), jqmc_ewald.grad_Eii_wrt_strain(cd, ed), atol, rtol
    )


# FAR-ROAMED e-i STRAIN (the strain twin of the e-i force row): ``grad_Eei_wrt_strain``
# must MIN-IMAGE ``r_i - R_I`` before the truncated ``T_n`` image sum, exactly as
# ``grad_Eee_wrt_strain`` and the ei energy already do. The near ``_cfg`` (scale 0.7,
# in-cell) never exposes it -- raw == min-image while the walker stays home. Once an electron
# roams an integer lattice vector out the raw ei-strain misses the dominant near image and
# diverges O(1) (the entire multi-Ha HF-stress divergence at a raw far-roamed walker).
@pytest.mark.light
@pytest.mark.parametrize("pattern", SYS)
@pytest.mark.parametrize("seed", [1, 3])
def test_ewald_ei_strain_far(pattern, seed):
    """Ewald ei strain far."""
    ham = rs_ham(pattern)
    cd, ed, eta, ri, gv, gf, cell, nmax, z, rion = _tabs(ham)
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    cellm = np.asarray(ham.structure_data.cell, float)  # rows = lattice vectors
    rng = np.random.default_rng(seed)

    def _roam(n):
        base = R[np.arange(n) % len(R)] + rng.normal(scale=0.4, size=(n, 3))
        return base + rng.integers(-2, 3, size=(n, 3)) @ cellm

    ru, rd = _roam(nu), _roam(nd)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONE)
    assert _close(
        jqmc_rs.grad_Eei_wrt_strain(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax),
        jqmc_ewald.grad_Eei_wrt_strain(cd, ed, ru, rd),
        atol,
        rtol,
    )

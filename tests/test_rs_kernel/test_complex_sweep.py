"""Rust vs XLA: native MCMC sweep for a COMPLEX Bloch twist geminal
(``_mcmc_sweep_complex``; fixtures are the committed hams).

The sweep scaffolding (threefry move draws, Kato drift proposal, T_ratio) is shared
with the real ``_mcmc_sweep``; the determinant ratio / G^-1 maintenance / AS
regularization are complex, and acceptance uses ``r_ratio = |R_AS.J.det_t|^2``.
Jastrow = J1/J2/DTN (real, geometry-only) or J3 (complex sphe, see
``test_complex_sweep_j3``).

Four kinds of checks:
  (a) STATE CONSISTENCY: post-sweep complex ``geminal_inv`` ~= inverse(G at swept
      config), and the emitted det-ratio state == jQMC ``_init_det_ratio_streaming_state``
      rebuilt there (proves the complex SM update + streaming advance, incl. the
      ComplexOrb far-field wrap+restore -- pass UNWRAPPED coords, no sweep-level
      pre-wrap).
  (b) ACCEPTANCE: accept count matches an independent complex per-move Python
      reference (same threefry/proposal/T_ratio, complex det ratio via
      ``compute_geminal_all_elements``, full Jastrow ratio, |.|^2 acceptance).
  (c) e_L-FAST CONSUMPTION: the Rust-emitted state feeds jQMC's
      ``compute_local_energy_fast`` byte-for-byte against the full
      ``compute_local_energy`` at the swept config (complex, incl. imaginary part).
  (d) NATIVE VMC: the whole complex chain (sweep + Re(complex e_L)) runs in Rust
      and its e_L assembly formula matches jQMC's ``compute_local_energy``.

Fixture note: the rows ``lithium_k025`` x {jnone,j12,jdtn} and a pgto(+)pw composite x
{jnone,j12} map onto 5 committed systems here -- ``twist_sphepgto_{jnone,j12,jdtn}`` (named
``_rs_fixtures.py`` patterns, exact match) and a genuine twist (k=0.25) QE pgto+pw
composite, ``{jnone,j12}`` (built via the ``_build_ham_for_cell`` escape hatch,
``QEPROJ:pgto+pw:Li_k025:ECP`` sentinel). See
``test_complex_composite.py`` for why the *named* ``pbc_composite_jnone``/``_j12``
patterns do NOT work here: that QE ``Li_gamma`` save is a genuine Gamma point, so
jqmc's builder takes the REAL plane-wave branch (no complex leaves anywhere in the
pytree) -- feeding it to the Rust ``*_complex`` family raises
``TypeError: 'ndarray' object cannot be converted to 'PyArray<T, D>'``. The
``QEPROJ:pgto+pw:Li_k025:ECP`` sentinel (same committed QE save the
``pw_twist_jnone``/``bspline_twist_jnone`` named patterns use, projected onto the
composite branch) is genuinely complex (``mo_coefficients.dtype == complex128``,
verified), which is what the composite rows need.
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

import jax.numpy as jnp
import jqmc_rs
import numpy as np
import pytest
from jax import random

from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance
from jqmc.determinant import (
    Det_ratio_streaming_state,
    _init_det_ratio_streaming_state,
    compute_geminal_all_elements,
)
from jqmc.hamiltonians import compute_local_energy, compute_local_energy_fast
from jqmc.jastrow_factor import _compute_ratio_Jastrow_part_rank1_update

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency

# Escape hatch (fixture requirement): no named ``_rs_fixtures.py``
# pattern covers a genuine twist composite, so it is built directly from the same
# single-source builder ``_rs_fixtures.rs_ham`` wraps (see module docstring).
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests._mcmc_kato_reference import nearest_charge_distance
from tests.test_rs_kernel._rs_fixtures import rs_ham

_COMPOSITE_TWIST_STEM = "QEPROJ:pgto+pw:Li_k025:ECP"
_COMPOSITE_JNONE = "composite_twist_jnone"
_COMPOSITE_J12 = "composite_twist_j12"

# pattern -> canonical committed-fixture name (see module docstring for the
# H8/H4/lithium_k025 -> 5-system collapse). The ``composite_twist_*`` names are
# NOT ``_rs_fixtures.py`` patterns (no ``rs_ham`` entry exists for them) -- they
# are resolved locally by ``_build_ham`` below via the escape hatch.
SYSTEMS = [
    "twist_sphepgto_jnone",
    "twist_sphepgto_j12",
    "twist_sphepgto_jdtn",
    _COMPOSITE_JNONE,
    _COMPOSITE_J12,
]
_DT = 1.2
_NMOVES = 20


def _build_ham(pattern):
    if pattern == _COMPOSITE_JNONE:
        return _build_ham_for_cell(_COMPOSITE_TWIST_STEM, False, None, "SD")
    if pattern == _COMPOSITE_J12:
        return _build_ham_for_cell(_COMPOSITE_TWIST_STEM, True, None, "SD")
    return rs_ham(pattern)


def _randomize_j3(jastrow_data, seed: int = 0, scale: float = 0.05):
    """Return jastrow_data with a non-zero random j_matrix.

    Committed j123ao fixtures may
    ship j_matrix == 0 (J3 inactive), which makes J3-in-sweep tests vacuous; this
    perturbs it so the complex-sphe J3 ratio path is actually exercised.
    """
    j3 = jastrow_data.jastrow_three_body_data
    if j3 is None:
        return jastrow_data
    rng = np.random.default_rng(seed)
    Jm = scale * rng.standard_normal(np.asarray(j3.j_matrix).shape)
    # Coupling block must be bosonic-symmetric (production sanity_check contract;
    # the native kernels exploit it) -- one-body column stays free.
    nao = Jm.shape[0]
    Jm[:, :nao] = 0.5 * (Jm[:, :nao] + Jm[:, :nao].T)
    return jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jnp.asarray(Jm)))


def _setup(pattern):
    ham = _build_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    cp = ham.coulomb_potential_data
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    cell = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.cell, dtype=np.float64)
    nmax = int(ham.structure_data._n_max)
    return ham, gd, jas, R, Z, cell, nmax


def _sweep(pattern, seed):
    ham, gd, jas, R, Z, cell, nmax = _setup(pattern)
    rng = np.random.default_rng(seed)
    ru0 = (rng.random((gd.num_electron_up, 3)) @ cell).astype(np.float64)
    rd0 = (rng.random((gd.num_electron_dn, 3)) @ cell).astype(np.float64)
    injected = rng.normal(size=_NMOVES).astype(np.float64)
    k = np.asarray(random.PRNGKey(seed), np.uint32)
    G0 = np.asarray(compute_geminal_all_elements(gd, ru0, rd0))
    res = jqmc_rs._mcmc_sweep_complex(
        gd, jas, R, Z, ru0, rd0, G0, np.linalg.inv(G0), _DT, 0.0, int(k[0]), int(k[1]), injected, cell, nmax
    )
    return ham, gd, jas, R, Z, ru0, rd0, injected, res


def _independent_acceptance_reference(gd, jas, R, Z, ru0, rd0, injected, seed, cell):
    """Per-move Python reference: same threefry/proposal/T_ratio, complex det ratio
    via ``compute_geminal_all_elements``, full Jastrow ratio, |.|^2 acceptance.
    Shared by the plain-Jastrow and J3-in-sweep acceptance tests."""

    def detc(ru, rd):
        return complex(np.linalg.det(np.asarray(compute_geminal_all_elements(gd, ru, rd))))

    def nearest(r):
        return nearest_charge_distance(R, Z, r, cell)

    ru, rd = ru0.copy(), rd0.copy()
    db = detc(ru, rd)
    key = random.PRNGKey(seed)
    nu_, nd_ = len(ru), len(rd)
    acc = 0
    for mv in range(_NMOVES):
        key, sk = random.split(key)
        rn = int(random.randint(sk, (), 0, nu_ + nd_))
        is_up = rn < nu_
        key, sk = random.split(key)
        ui = int(random.randint(sk, (), 0, nu_))
        key, sk = random.split(key)
        di = int(random.randint(sk, (), 0, nd_))
        sel = ui if is_up else di
        old = (ru if is_up else rd)[sel]
        z, nm = nearest(old)
        fl = 1.0 / z**2 * (1 + z**2 * nm) / (1 + nm)
        key, sk = random.split(key)
        g = float(injected[mv]) * (fl * _DT)
        key, sk = random.split(key)
        ax = int(random.randint(sk, (), 0, 3))
        new = old.copy()
        new[ax] += g
        z2, n2 = nearest(new)
        fp = 1.0 / z2**2 * (1 + z2**2 * n2) / (1 + n2)
        d2 = float(((new - old) ** 2).sum())
        tr = (fl / fp) * np.exp(-d2 * (1 / (2 * fp**2 * _DT**2) - 1 / (2 * fl**2 * _DT**2)))
        nu, nd = ru.copy(), rd.copy()
        if is_up:
            nu[sel] = new
        else:
            nd[sel] = new
        det_t = detc(nu, nd) / db
        jr = float(np.asarray(_compute_ratio_Jastrow_part_rank1_update(jas, ru, rd, nu[None], nd[None]))[0])
        rr = abs(jr * det_t) ** 2
        key, sk = random.split(key)
        b = float(random.uniform(sk, (), dtype=jnp.float64))
        if b < min(1.0, rr * tr):
            ru, rd, db = nu, nd, detc(nu, nd)
            acc += 1
    return acc


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", SYSTEMS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complex_sweep_state_consistency(pattern, seed):
    """Complex sweep state consistency."""
    _, gd, _, _, _, _, _, _, res = _sweep(pattern, seed)
    atol_g, rtol_g = get_tolerance_xla_rust_consistency(("det_ratio", "det_eval"), level="medium")
    atol_s, rtol_s = _d8_backend_tolerance(("det_ratio", "ao_eval"), level="medium")
    Gf = np.asarray(compute_geminal_all_elements(gd, res["r_up"], res["r_dn"]))
    gi_fresh = np.linalg.inv(Gf)
    rel = np.abs(np.asarray(res["geminal_inv"]) - gi_fresh).max() / (np.abs(gi_fresh).max() + 1e-300)
    assert rel <= atol_g + rtol_g, (pattern, seed, rel)
    st = _init_det_ratio_streaming_state(gd, res["r_up"], res["r_dn"])
    for f in ("ao_up", "ao_dn", "paired_dn", "paired_up_lambda"):
        ref = np.asarray(getattr(st, f))
        got = np.asarray(res[f])
        d = np.abs(got - ref).max()
        assert d <= atol_s + rtol_s * np.abs(ref).max(), (pattern, seed, f, d)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", SYSTEMS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complex_sweep_acceptance(pattern, seed):
    """Accept count matches an independent complex per-move reference. Accept counts
    are integers, so exact equality (no tolerance) is the correct comparison."""
    ham, gd, jas, R, Z, ru0, rd0, injected, res = _sweep(pattern, seed)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    acc = _independent_acceptance_reference(gd, jas, R, Z, ru0, rd0, injected, seed, cell)
    assert res["accepted"] == acc, (pattern, seed, res["accepted"], acc)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern",
    ["twist_sphepgto_jnone", _COMPOSITE_JNONE],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_complex_sweep_e_L_fast(pattern, seed):
    """Sweep-state contract (complex): the complex sweep emits complex128 geminal_inv +
    Det_ratio_streaming_state that jQMC's ``compute_local_energy_fast``
    (``_e_L_fast``) consumes -- the COMPLEX byte-match contract. Fed the
    Rust-emitted state, ``_e_L_fast`` == the full ``compute_local_energy``
    (complex, incl. the imaginary part) at the swept config. n=1 (twist lithium)
    + n=1 composite (committed twist pgto+pw substitute -- see module docstring)."""
    ham, gd, jas, R, Z, ru0, rd0, injected, res = _sweep(pattern, seed)
    # ``ao_eval`` is crossed by the non-local ECP term of e_L (its imaginary part
    # included), which evaluates AOs at the quadrature points in the ao_eval zone.
    # The zone is declared here.
    # Strict level: with ao_eval the medium envelope would widen the mixed budget
    # ~1000x (1e-4, 2e-3); strict (2e-7, 2e-6) covers the fp32 ao_eval term only.
    atol, rtol = _d8_backend_tolerance(("local_energy", "det_ratio", "wf_kinetic", "coulomb", "ao_eval"), level="strict")
    st = Det_ratio_streaming_state(
        ao_up=jnp.asarray(res["ao_up"]),
        ao_dn=jnp.asarray(res["ao_dn"]),
        paired_dn=jnp.asarray(res["paired_dn"]),
        paired_up_lambda=jnp.asarray(res["paired_up_lambda"]),
    )
    ef = complex(compute_local_energy_fast(ham, res["r_up"], res["r_dn"], jnp.eye(3), jnp.asarray(res["geminal_inv"]), st))
    es = complex(compute_local_energy(ham, res["r_up"], res["r_dn"], jnp.eye(3)))
    # Inline |ef - es| (no alias that may differ by one ulp from the receipt's replay).
    assert abs(ef - es) <= atol + rtol * abs(es), (pattern, seed, ef, es, abs(ef - es))


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complex_sweep_as_regularization(seed):
    """Complex AS regularization (epsilon_as>0) in the sweep acceptance. A huge
    epsilon_as forces ``max(R_AS,eps)=eps``, so ``R_AS_ratio = R_AS_old/R_AS_proposed``
    is fully active -- this validates ``compute_as_regularization_factor_complex``
    (the materialized-proposed-G^-1 path) against a Python AS-factor reference.

    Fixture note: the committed composite (the genuine twist QE pgto+pw composite,
    ``composite_twist_jnone``) is 1x1 (Li, single electron per spin), so the
    AS-regularization *row count* is smaller here, but the exercised Rust code
    path (``compute_as_regularization_factor_complex``, the
    materialized-proposed-G^-1 branch) is identical.
    """
    pattern = _COMPOSITE_JNONE
    ham, gd, jas, R, Z, cell, nmax = _setup(pattern)
    eps = 1e30  # forces the max() branch -> R_AS_ratio = R_AS_o / R_AS_p
    rng = np.random.default_rng(seed)
    ru0 = (rng.random((gd.num_electron_up, 3)) @ cell).astype(np.float64)
    rd0 = (rng.random((gd.num_electron_dn, 3)) @ cell).astype(np.float64)
    injected = rng.normal(size=_NMOVES).astype(np.float64)
    k = np.asarray(random.PRNGKey(seed), np.uint32)
    G0 = np.asarray(compute_geminal_all_elements(gd, ru0, rd0))
    res = jqmc_rs._mcmc_sweep_complex(
        gd, jas, R, Z, ru0, rd0, G0, np.linalg.inv(G0), _DT, eps, int(k[0]), int(k[1]), injected, cell, nmax
    )

    def gmat(ru, rd):
        return np.asarray(compute_geminal_all_elements(gd, ru, rd))

    def as_factor(G):
        Gi = np.linalg.inv(G)
        F = (np.abs(Gi) ** 2).sum()
        S = min((np.abs(G) ** 2).sum(1).min(), (np.abs(G) ** 2).sum(0).min())
        sf = S * F
        return sf ** (-3.0 / 8.0) if sf > 0 else 0.0

    def nearest(r):
        return nearest_charge_distance(R, Z, r, cell)

    ru, rd = ru0.copy(), rd0.copy()
    G = gmat(ru, rd)
    key = random.PRNGKey(seed)
    nu_, nd_ = len(ru), len(rd)
    acc = 0
    for mv in range(_NMOVES):
        key, sk = random.split(key)
        is_up = int(random.randint(sk, (), 0, nu_ + nd_)) < nu_
        key, sk = random.split(key)
        ui = int(random.randint(sk, (), 0, nu_))
        key, sk = random.split(key)
        di = int(random.randint(sk, (), 0, nd_))
        sel = ui if is_up else di
        old = (ru if is_up else rd)[sel]
        z, nm = nearest(old)
        fl = 1.0 / z**2 * (1 + z**2 * nm) / (1 + nm)
        key, sk = random.split(key)
        g = float(injected[mv]) * (fl * _DT)
        key, sk = random.split(key)
        ax = int(random.randint(sk, (), 0, 3))
        new = old.copy()
        new[ax] += g
        z2, n2 = nearest(new)
        fp = 1.0 / z2**2 * (1 + z2**2 * n2) / (1 + n2)
        d2 = float(((new - old) ** 2).sum())
        tr = (fl / fp) * np.exp(-d2 * (1 / (2 * fp**2 * _DT**2) - 1 / (2 * fl**2 * _DT**2)))
        nu, nd = ru.copy(), rd.copy()
        if is_up:
            nu[sel] = new
        else:
            nd[sel] = new
        Gn = gmat(nu, nd)
        det_t = np.linalg.det(Gn) / np.linalg.det(G)
        r_as_ratio = as_factor(G) / as_factor(Gn)  # huge eps -> max()=eps both
        key, sk = random.split(key)
        b = float(random.uniform(sk, (), dtype=jnp.float64))
        if b < min(1.0, abs(r_as_ratio * det_t) ** 2 * tr):
            ru, rd, G = nu, nd, Gn
            acc += 1
    assert res["accepted"] == acc, (seed, res["accepted"], acc)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern",
    ["twist_sphepgto_jnone", "twist_sphepgto_j12"],
)
def test_native_vmc_complex(pattern):
    """Complex-twist native VMC chain (``native_vmc_complex``): the whole VMC loop
    (complex sweep + Re(complex e_L)) runs in Rust. (a) it produces sensible
    per-block energies; (b) its e_L formula
    Re(T_complex)+Ewald+ECP-local+Re(ECP-nl-complex) == Re(jQMC compute_local_energy),
    verified by assembling the same standalone Rust complex wrappers the driver
    uses."""
    from jqmc.coulomb_potential import (
        compute_ecp_local_parts_all_pairs,
        compute_ewald_potential_ee,
        compute_ewald_potential_ei,
        icosahedron_sym_mesh_Nv12,
    )

    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
    ham = rs_ham(pattern)
    atol, rtol = _d8_backend_tolerance(
        ("local_energy", "wf_kinetic", "coulomb", "ewald", "ao_eval", "jastrow_grad_lap"), level="medium"
    )

    # (a) the chain runs and yields finite energies
    en, acc = jqmc_rs.native_vmc_complex(ham, w, gp, 7, 1.0, 0.0, 40, 2, 4)
    assert len(en) == 4 and np.all(np.isfinite(en)) and 0.0 <= acc <= 1.0

    # (b) e_L assembly (the driver's formula) == Re(compute_local_energy) at fixed configs
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        rng = np.random.default_rng(seed)
        ru = R[: gd.num_electron_up] + rng.normal(scale=0.6, size=(gd.num_electron_up, 3))
        rd = R[: gd.num_electron_dn] + rng.normal(scale=0.6, size=(gd.num_electron_dn, 3))
        tre, _ = jqmc_rs.compute_kinetic_withj_complex(gd, jd, ru, rd)
        vnl_re, _ = jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, jd, cd, ru, rd, w, gp)
        ei_t = compute_ewald_potential_ei(cd, ru, rd, ed)
        ei = float(np.asarray(ei_t[0]).sum() + np.asarray(ei_t[1]).sum())
        v = (
            float(compute_ewald_potential_ee(ru, rd, ed))
            + ei
            + float(ed.ii_energy)
            + float(compute_ecp_local_parts_all_pairs(cd, ru, rd))
        )
        assembled = tre + v + vnl_re
        ref = complex(compute_local_energy(ham, ru, rd, jnp.eye(3))).real
        d = abs(assembled - ref)
        assert d <= atol + rtol * abs(ref), (pattern, seed, assembled, ref, d)


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complex_sweep_j3(seed):
    """Complex-Bloch J3 (A = Re(complex sphe AO)) in the per-move sweep ratio. The
    committed twist+J3 ham (built via ``_build_ham_for_cell``; the same recipe as the
    named ``twist_sphepgto_j123`` pattern in ``_rs_fixtures``) is further randomized
    (``_randomize_j3``) to guarantee J3 is truly active regardless of the shipped
    j_matrix. Accept count vs the complex per-move reference (full Jastrow incl J3)
    + det-state byte-match."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    ham = _build_ham_for_cell("lithium_k025.h5", True, {"j3": "pgto"}, "SD")
    gd = ham.wavefunction_data.geminal_data
    jas0 = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    cp = ham.coulomb_potential_data
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    cell = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.cell, dtype=np.float64)
    nmax = int(ham.structure_data._n_max)

    jas = _randomize_j3(jas0, seed=7)
    rng = np.random.default_rng(seed)
    ru0 = (rng.random((gd.num_electron_up, 3)) @ cell).astype(np.float64)
    rd0 = (rng.random((gd.num_electron_dn, 3)) @ cell).astype(np.float64)
    injected = rng.normal(size=_NMOVES).astype(np.float64)
    k = np.asarray(random.PRNGKey(seed), np.uint32)
    G0 = np.asarray(compute_geminal_all_elements(gd, ru0, rd0))
    res = jqmc_rs._mcmc_sweep_complex(
        gd, jas, R, Z, ru0, rd0, G0, np.linalg.inv(G0), _DT, 0.0, int(k[0]), int(k[1]), injected, cell, nmax
    )

    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    acc = _independent_acceptance_reference(gd, jas, R, Z, ru0, rd0, injected, seed, cell)
    assert res["accepted"] == acc, (seed, res["accepted"], acc)

    atol, rtol = _d8_backend_tolerance(("det_ratio", "ao_eval", "jastrow_eval"), level="medium")
    st = _init_det_ratio_streaming_state(gd, res["r_up"], res["r_dn"])
    for f in ("ao_up", "ao_dn", "paired_dn", "paired_up_lambda"):
        ref = np.asarray(getattr(st, f))
        got = np.asarray(res[f])
        d = np.abs(got - ref).max()
        assert d <= atol + rtol * np.abs(ref).max(), (seed, f, d)

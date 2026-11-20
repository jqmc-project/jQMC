"""Rust vs XLA: complex composite pgto (+) pw leaf (PBC twist k != Gamma code path;
on a committed QE twist composite).

The plane-wave child is a pure complex exponential ``AO_k(r) = e^{i.gk.r}``
(``gk = G + k``, ``_gk_vectors_jnp``, no cell wrap so the Bloch phase survives);
grad = i.gk.AO, lap = -|gk|^2.AO. ``ComplexOrb`` concatenates the complex gto
block (Bloch image-phase fold) and the pw block in the leaf's gto-first ordering,
then applies the complex MO.

Note on fixture choice (IMPORTANT): the named ``_rs_fixtures.py`` patterns ``pbc_composite_jnone`` /
``pbc_composite_j12`` (QE ``Li_gamma``, a genuine Gamma-point QE save) were
suspected to be a valid substitute on the theory that the plane-wave leaf is
complex "from G alone" even at k=0 -- but this does NOT hold for jqmc's actual
builder: at a genuine QE Gamma save, ``qe_to_jqmc.py`` takes the REAL branch
(``AOs_plane_wave_data``, real cos/sin(G.r) basis) rather than
``AOs_plane_wave_complex_data``, so ``pbc_composite_jnone``/``_j12`` are entirely
real float64 (verified: every leaf of the ``aos_data`` pytree, incl.
``mo_coefficients``, is float64) -- feeding them to the Rust ``*_complex`` family
(``ComplexOrb::from_py`` extracts ``PyReadonlyArray2<Complex64>``, no implicit
real->complex cast) raises ``TypeError: 'ndarray' object cannot be converted to
'PyArray<T, D>'``. The correct committed substitute is the genuine twist (k=0.25)
QE composite, reached via the ``_build_ham_for_cell`` escape hatch with the
``QEPROJ:pgto+pw:Li_k025:ECP`` sentinel (same committed ``tests/qe_example_files/
Li_k025`` save the ``pw_twist_jnone`` / ``bspline_twist_jnone`` named patterns use,
just projected onto the pgto+pw composite branch instead of pure-pw/B-spline) --
verified live: ``mo_coefficients.dtype == complex128`` and
``jqmc_rs.compute_ln_det_complex`` succeeds. This is not a named pattern in
``_rs_fixtures.py`` (composite-twist has no named pattern there), so it is built
directly here via the documented escape hatch.

Verified vs jqmc: ln|det|, kinetic energy (complex), O_k=d ln|det|/d MO, dT/dMO,
ECP non-local (Re). All rows use the Jastrow-none variant: ``compute_kinetic_det_complex``
/ ``grad_T_mo_complex`` are DETERMINANT-ONLY kinetic kernels (no Jastrow term), so
they are only comparable against jqmc's ``compute_kinetic_energy`` when the
wavefunction's Jastrow is inactive.

NOTE: physical-config accuracy only (electrons near atoms). Far-field (|r| >~
half-cell) needs the jqmc wrap+restore image convention; the ECP non-local mesh
probes that region and is a separate item (complex-gto far-field, not pw-specific).
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
import jax.numpy as jnp
import jqmc_rs
import numpy as np
import pytest

from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance
from jqmc.determinant import compute_ln_det_geminal_all_elements as LD
from jqmc.wavefunction import compute_kinetic_energy as CK

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency

# Escape hatch (fixture requirement): no named ``_rs_fixtures.py``
# pattern covers a genuine twist composite, so we build directly from the same
# single-source builder ``_rs_fixtures.rs_ham`` wraps.
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

# Committed genuine twist (k=0.25) QE pgto+pw composite, ECP (see module docstring
# for why the real-Gamma ``pbc_composite_*`` patterns do NOT work here).
_COMPOSITE_TWIST_STEM = "QEPROJ:pgto+pw:Li_k025:ECP"


def _composite_ham():
    return _build_ham_for_cell(_COMPOSITE_TWIST_STEM, False, None, "SD")


def _cfg(ham, seed):
    gd = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    nu, ndn = gd.num_electron_up, gd.num_electron_dn
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.6, size=(nu, 3))
    rd = R[np.arange(ndn) % len(R)] + rng.normal(scale=0.6, size=(ndn, 3))
    return ru, rd


@pytest.mark.heavy
def test_composite_lndet_and_kinetic():
    """Composite lndet and kinetic."""
    ham = _composite_ham()
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    atol_ld, rtol_ld = get_tolerance_xla_rust_consistency(("det_eval", "ao_eval"), level="strict")
    atol_k, rtol_k = _d8_backend_tolerance(("wf_kinetic", "ao_eval", "ao_grad_lap"), level="strict")
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        ld_ref = float(np.real(LD(gd, ru, rd)))
        ld_got = jqmc_rs.compute_ln_det_complex(gd, ru, rd)
        assert abs(ld_got - ld_ref) <= atol_ld + rtol_ld * abs(ld_ref), (seed, ld_got, ld_ref)
        re, im = jqmc_rs.compute_kinetic_det_complex(gd, ru, rd)
        k_ref = complex(CK(wf, ru, rd))
        k_got = complex(re, im)
        d = abs(k_got - k_ref)
        assert d <= atol_k + rtol_k * abs(k_ref), (seed, k_got, k_ref, d)


@pytest.mark.heavy
def test_composite_Ok_and_dT_dMO():
    """O_k = d ln|det|/d MO and dT/dMO over the FULL composite MO (gto (+) pw columns).

    Jastrow-none only: ``grad_T_mo_complex`` is the determinant-only kinetic
    gradient (no Jastrow term), so it is only comparable against jax.grad of the
    full ``compute_kinetic_energy`` when the Jastrow is inactive."""
    ham = _composite_ham()
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    atol_o, rtol_o = get_tolerance_xla_rust_consistency(("ao_eval", "mo_grad", "det_grad_lap"), level="medium")
    atol_t, rtol_t = get_tolerance_xla_rust_consistency(("ao_eval", "mo_grad", "det_grad_lap", "wf_kinetic"), level="medium")
    for seed in range(2):
        ru, rd = _cfg(ham, seed)
        # Rust BEFORE jax (jit tracing of the composite pytree can perturb later reads).
        ou, od = [np.asarray(x) for x in jqmc_rs.grad_ln_det_mo_complex(gd, ru, rd)]
        tu, td = [np.asarray(x) for x in jqmc_rs.grad_T_mo_complex(gd, ru, rd)]
        gO = jax.grad(lambda G: LD(G, ru, rd).real)(gd)
        gT = jax.grad(lambda G: CK(wf.replace(geminal_data=G), ru, rd).real)(gd)
        ref_ou = np.asarray(gO.orb_data_up_spin.mo_coefficients)
        ref_od = np.asarray(gO.orb_data_dn_spin.mo_coefficients)
        ref_tu = np.asarray(gT.orb_data_up_spin.mo_coefficients)
        ref_td = np.asarray(gT.orb_data_dn_spin.mo_coefficients)
        assert np.abs(ou - ref_ou).max() <= atol_o + rtol_o * np.abs(ref_ou).max()
        assert np.abs(od - ref_od).max() <= atol_o + rtol_o * np.abs(ref_od).max()
        assert np.abs(tu - ref_tu).max() <= atol_t + rtol_t * np.abs(ref_tu).max()
        assert np.abs(td - ref_td).max() <= atol_t + rtol_t * np.abs(ref_td).max()


@pytest.mark.heavy
@pytest.mark.parametrize("scale", [3.0, 6.0])
def test_composite_farfield_lndet(scale):
    """The periodic wrap+restore makes the complex gto robust for |r| >~ half-cell
    (electron crosses cell faces). Without it the static image-tiled kernel misses
    images and ln|det| diverges. Verified at increasing scales."""
    ham = _composite_ham()
    gd = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    nu, ndn = gd.num_electron_up, gd.num_electron_dn
    atol, rtol = get_tolerance_xla_rust_consistency(("det_eval", "ao_eval"), level="strict")
    rng = np.random.default_rng(7)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[np.arange(ndn) % len(R)] + rng.normal(scale=scale, size=(ndn, 3))
    ref = float(np.real(LD(gd, ru, rd)))
    got = jqmc_rs.compute_ln_det_complex(gd, ru, rd)
    assert abs(got - ref) <= atol + rtol * abs(ref), (scale, got, ref)


@pytest.mark.heavy
def test_composite_ecp_nonlocal():
    """Complex ECP non-local energy (Re; jqmc takes Re(P_l.ratio) per grid term) for
    the committed pgto (+) pw composite. The quadrature mesh r'=R_I+r_iI.v_g lands
    outside the home cell, so this needs the gto wrap+restore (was off without it)."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors as ECPNL
    from jqmc.coulomb_potential import octahedron_sym_mesh_Nv6 as MESH

    ham = _composite_ham()
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    cd = ham.coulomb_potential_data
    w = np.asarray(MESH.weights, dtype=np.float64)
    gp = np.asarray(MESH.grid_points, dtype=np.float64)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    natom = R.shape[0]
    atol, rtol = get_tolerance_xla_rust_consistency(("coulomb", "ao_eval", "det_eval"), level="strict")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        ru = R[:1] + rng.normal(scale=0.7, size=(1, 3))
        rd = R[:1] + rng.normal(scale=0.7, size=(1, 3))
        ref = float(np.real(ECPNL(cd, wf, ru, rd, jnp.eye(3), NN=natom, Nv=6, flag_determinant_only=True)[3]))
        re, im = jqmc_rs.compute_ecp_nonlocal_complex(gd, cd, ru, rd, w, gp)
        assert abs(re - ref) <= atol + rtol * abs(ref), (seed, re, ref)

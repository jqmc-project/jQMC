"""Layer-1: complex Bloch core (PBC twist k!=Gamma) -- geminal/ln|det|, determinant
kinetic energy (complex), ECP non-local energy (real per jqmc convention), and the
full complex-twist gradient stack (O_k/force/stress, withJ J1/J2/J3, DTN) vs
jqmc's own XLA analytic / autodiff oracles.

The complex local energy is e_L = T(complex) + V(real): e_L's imaginary part = the
kinetic imaginary part (Ewald + ECP-local + ECP-non-local are all real).

Every ham is built from COMMITTED fixtures (``tests/trexio_example_files/lithium_k025.h5`` complex
twist k=0.25 + QE ``Li_gamma`` pgto(+)pw composite) via the shared single-source
builders in ``_rs_fixtures`` / ``_generated_matrix_base``. A Rust==XLA
equivalence is code-path sensitive (not system sensitive), so the committed twist cell
flows the identical complex Bloch code path any twist cell does.

Two in-file helpers:
  * the ``_randomize_j3`` / ``_randomize_dtn`` helpers below (the committed j123ao / jdtn
    recipes ship near-zero exotic params, so the tests randomize them to actually exercise
    the J3 / DTN-f channel).
  * a direct ``jax.jacrev(compute_kinetic_energy o proper-strain)``
    oracle (``_kinetic_strain_withJ_oracle``), using the production strain recipe the file's
    other withJ-strain tests use (``ham._apply_strain``, which strains the J2 cell with
    the physical lattice).

The committed twist cell has NO composite-twist analogue, so the Li2
pgto(+)pw twist rows run on the real-Gamma composite via the escape-hatch builder. The
complex kernel accepts a REAL-Gamma plane-wave child (``AOs_plane_wave_data``, k = 0:
``gk == G``), so feeding it the real-Gamma composite is a legitimate k = 0 cross-check --
the complex kernel must reduce to the real result (imaginary parts vanishing). Those rows
are HARD-asserted against the jqmc oracle (no self-skip).
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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance  # noqa: E402
from jqmc._setting import (  # noqa: E402
    NN_default,
    RTOL_FD_VS_ANALYTIC_TIGHT,
)
from jqmc.coulomb_potential import (  # noqa: E402
    compute_ecp_non_local_parts_nearest_neighbors,
    icosahedron_sym_mesh_Nv12,
    octahedron_sym_mesh_Nv6,
)
from jqmc.determinant import (  # noqa: E402
    compute_geminal_all_elements,
    compute_ln_det_geminal_all_elements,
)
from jqmc.jqmc_mcmc import _inject_active_positions  # noqa: E402
from jqmc.wavefunction import compute_kinetic_energy  # noqa: E402

get_tolerance_xla_rust_consistency = functools.partial(
    get_tolerance_xla_rust_consistency,
    level="medium",
)

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

# Mandatory CWD classification (tests/conftest.py): every ham loads from an ABSOLUTE
# committed-fixture path, so the autouse cwd isolation is harmless -> with_chdir.
pytestmark = pytest.mark.with_chdir

_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Committed-fixture ham builders (built once, reused -- mirrors how
# test_gfmc_complex_force_stress.py builds per Jastrow variant).
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _pat(pattern: str):
    """Cached named-pattern ham (twist_sphepgto_{jnone,j12,jdtn})."""
    return rs_ham(pattern)


@functools.lru_cache(maxsize=None)
def _cell(stem: str, with_jastrow: bool, spec_key: tuple, gem: str):
    """Cached escape-hatch ham for combos with no named pattern (e.g. twist + J3)."""
    spec = dict(spec_key) if spec_key else None
    return _build_ham_for_cell(stem, with_jastrow, spec, gem)


def _randomize_j3(jastrow_data, seed: int = 0, scale: float = 0.05):
    """Return jastrow_data with a non-zero random j_matrix (committed j123ao fixtures/specs
    may ship j_matrix == 0, which makes J3 gradient tests vacuous)."""
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


def _randomize_dtn(jastrow_data, seed: int = 0, scale: float = 0.05):
    """Return jastrow_data with non-zero random DTN alpha/beta/gamma_indep (committed jdtn
    fixtures may ship all-zero indep params, making DTN value-derivative tests vacuous)."""
    d = jastrow_data.jastrow_dtn_data
    if d is None:
        return jastrow_data
    rng = np.random.default_rng(seed)
    upd = {}
    for f in ("alpha_indep", "beta_indep", "gamma_indep"):
        v = np.asarray(getattr(d, f))
        upd[f] = jnp.asarray(scale * rng.standard_normal(v.shape))
    return jastrow_data.replace(jastrow_dtn_data=d.replace(**upd))


def _cfg(ham, seed):
    """Draw an electron config near the ions (scale 0.7)."""
    gd = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[: gd.num_electron_up] + rng.normal(scale=0.7, size=(gd.num_electron_up, 3))
    rd = R[: gd.num_electron_dn] + rng.normal(scale=0.7, size=(gd.num_electron_dn, 3))
    return ru, rd


def _kinetic_strain_withJ_oracle(ham, ru, rd):
    """Direct jax.jacrev oracle for the withJ kinetic STRESS dT/deps. Uses the SAME proper-strain
    recipe the file's other withJ-strain tests use: F = I + eps applied to the whole wf
    via ``ham._apply_strain`` (strains geminal + J1/J2/J3/DTN cells), electrons mapped
    r -> r @ F.T. This follows the current production definition directly."""

    def Ts(eps):
        F = jnp.eye(3) + eps
        ham2 = ham._apply_strain(F)
        wf2 = ham2.wavefunction_data
        return jnp.real(compute_kinetic_energy(wf2, ru @ F.T, rd @ F.T))

    return np.asarray(jax.jacrev(Ts)(jnp.zeros((3, 3))))


# The 3-row withJ ECP loop: j12 + twist-j123ao(rand) + composite-j123ao(rand). The
# composite row feeds a REAL-Gamma pgto(+)pw child into the complex kernel: a legitimate
# k = 0 cross-check (the complex kernel must reduce to the real result, imaginary parts
# vanishing). All three rows HARD-assert against the jqmc oracle.
_WITHJ_ECP_ROWS = ["j12", "j123ao", "composite_j123ao"]


def _withj_ecp_row(row: str):
    """Build (ham, gd, jd, cd, natom) for one withJ-ECP loop row."""
    if row == "j12":
        ham = _pat("twist_sphepgto_j12")
        jd = ham.wavefunction_data.jastrow_data
    elif row == "j123ao":
        ham = _cell("lithium_k025.h5", True, (("j3", "pgto"),), "SD")
        jd = _randomize_j3(ham.wavefunction_data.jastrow_data, seed=7)
    elif row == "composite_j123ao":
        ham = _cell("QE:Li_gamma_pgto+pw", True, (("j3", "mo"),), "SD")
        jd = _randomize_j3(ham.wavefunction_data.jastrow_data, seed=7)
    else:  # pragma: no cover
        raise ValueError(row)
    gd = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    natom = np.asarray(ham.structure_data.positions).shape[0]
    return ham, gd, jd, cd, natom


# --------------------------------------------------------------------------- #
# jnone value checks (light)
# --------------------------------------------------------------------------- #
@pytest.mark.light
def test_geminal_and_lndet_complex():
    """Geminal and lndet complex."""
    ham = _pat("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_eval"), level="strict")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        Gref = np.asarray(compute_geminal_all_elements(gd, ru, rd))
        Ggot = np.asarray(jqmc_rs.compute_geminal_complex(gd, ru, rd))
        np.testing.assert_allclose(Ggot, Gref, atol=atol, rtol=rtol, err_msg=f"geminal seed={seed}")
        lref = float(np.asarray(compute_ln_det_geminal_all_elements(gd, ru, rd)))
        lgot = float(jqmc_rs.compute_ln_det_complex(gd, ru, rd))
        np.testing.assert_allclose(lgot, lref, atol=atol, rtol=rtol, err_msg=f"lndet seed={seed}")


@pytest.mark.light
def test_kinetic_complex():
    """Kinetic complex."""
    ham = _pat("twist_sphepgto_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    atol, rtol = _d8_backend_tolerance(("ao_eval", "ao_grad_lap", "mo_grad", "wf_kinetic", "det_eval"), level="strict")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        tref = complex(compute_kinetic_energy(wf, ru, rd))
        re, im = jqmc_rs.compute_kinetic_det_complex(gd, ru, rd)
        np.testing.assert_allclose(complex(re, im), tref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


@pytest.mark.light
def test_ecp_nonlocal_complex():
    """ECP non-local energy (real part = jqmc's value; jqmc returns Re of the
    complex det-ratio sum). NN=natom disables the nearest-neighbour cutoff."""
    ham = _pat("twist_sphepgto_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    cd = ham.coulomb_potential_data
    natom = np.asarray(ham.structure_data.positions).shape[0]
    atol, rtol = _d8_backend_tolerance(("coulomb",), level="strict")
    weights = np.asarray(octahedron_sym_mesh_Nv6.weights, dtype=np.float64)
    grid_points = np.asarray(octahedron_sym_mesh_Nv6.grid_points, dtype=np.float64)
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        ref = complex(
            compute_ecp_non_local_parts_nearest_neighbors(
                cd,
                wf,
                ru,
                rd,
                jnp.eye(3),
                NN=natom,
                Nv=6,
                flag_determinant_only=True,
            )[3]
        )
        re, im = jqmc_rs.compute_ecp_nonlocal_complex(gd, cd, ru, rd, weights, grid_points)
        np.testing.assert_allclose(re, ref.real, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


# --------------------------------------------------------------------------- #
# jnone gradient stack (heavy)
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_grad_ln_det_complex_Ok():
    """Complex SD O_k = dln|det|/dMO (holomorphic) vs JAX autograd.

    Canonical ``SD_data`` owns occupied orbitals only and has no lambda
    coordinate.  Complex lambda-kernel parity is covered independently with a
    true AGP fixture by ``test_ungate_complex_twist_param_deriv.py``.
    """
    ham = _pat("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    atol, rtol = _d8_backend_tolerance(("ao_eval", "mo_grad", "det_eval"), level="medium")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        g = jax.grad(lambda G: jnp.real(compute_ln_det_geminal_all_elements(G, ru, rd)))(gd)
        mu, md = [np.asarray(a) for a in jqmc_rs.grad_ln_det_mo_complex(gd, ru, rd)]
        np.testing.assert_allclose(
            mu, np.asarray(g.orb_data_up_spin.mo_coefficients), atol=atol, rtol=rtol, err_msg=f"mo_up seed={seed}"
        )
        np.testing.assert_allclose(
            md, np.asarray(g.orb_data_dn_spin.mo_coefficients), atol=atol, rtol=rtol, err_msg=f"mo_dn seed={seed}"
        )


@pytest.mark.heavy
def test_grad_lndet_R_complex_G3():
    """Complex Pulay force dln|det|/dR for the twisted geminal: translation
    identity dAO/dR=-grad AO (Bloch phase preserved), Re of the complex contraction.
    Verified vs jax.grad(Re ln det) wrt atomic positions."""
    ham = _pat("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    pos = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.positions)
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_grad_lap", "det_eval"), level="medium")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        ref = np.asarray(
            jax.grad(
                lambda P: jnp.real(compute_ln_det_geminal_all_elements(_inject_active_positions(gd, {"positions": P}), ru, rd))
            )(pos)
        )
        got = np.asarray(jqmc_rs.grad_lndet_wrt_R_complex(gd, ru, rd))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


def _strain_oracle(ham, gd, ru, rd, fn):
    sd = ham.structure_data
    va, vb, vc = jnp.asarray(sd.vec_a), jnp.asarray(sd.vec_b), jnp.asarray(sd.vec_c)
    pos = jnp.asarray(np.asarray(sd.positions, float))

    def f(eps):
        F = jnp.eye(3) + eps
        sd2 = sd.replace(vec_a=F @ va, vec_b=F @ vb, vec_c=F @ vc, positions=pos @ F.T)
        repl = lambda od: (
            od.replace(aos_data=od.aos_data.replace(structure_data=sd2))
            if type(od).__name__ == "MOs_data"
            else od.replace(structure_data=sd2)
        )
        gd2 = gd.replace(orb_data_up_spin=repl(gd.orb_data_up_spin), orb_data_dn_spin=repl(gd.orb_data_dn_spin))
        return jnp.real(fn(gd2, ru @ F.T, rd @ F.T))

    return np.asarray(jax.grad(f)(jnp.zeros((3, 3))))


@pytest.mark.heavy
def test_grad_lnPsi_strain_complex_G5():
    """Complex Pulay stress dln|det|/deps for the twisted geminal (complex
    field_stress: affine virial + image-T, sphe rank, Bloch phase; unwrapped r -> no
    restore). Verified vs jax.grad(Re ln det) wrt strain."""
    ham = _pat("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_grad_lap", "swct", "det_eval"), level="medium")
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        ref = _strain_oracle(ham, gd, ru, rd, lambda G, a, b: compute_ln_det_geminal_all_elements(G, a, b))
        got = np.asarray(jqmc_rs.grad_lnPsi_wrt_strain_complex(gd, ham.wavefunction_data.jastrow_data, ru, rd))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


@pytest.mark.heavy
def test_grad_T_strain_complex_G6():
    """Complex kinetic stress dT/deps for the twisted geminal, jnone (complex
    geminal_reverse cotangents + value/gradlap field strain). Verified vs
    jax.grad(Re compute_kinetic_energy) wrt strain."""
    ham = _pat("twist_sphepgto_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    atol, rtol = _d8_backend_tolerance(("ao_eval", "wf_kinetic", "swct", "mo_grad"), level="medium")
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        ref = _strain_oracle(ham, gd, ru, rd, lambda G, a, b: compute_kinetic_energy(wf.replace(geminal_data=G), a, b))
        got = np.asarray(jqmc_rs.grad_T_wrt_strain_complex(gd, ru, rd))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


# --------------------------------------------------------------------------- #
# FULL complex ECP local-energy stress & force (heavy)
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_grad_eL_strain_complex_ECP_full():
    """FULL complex ECP local-energy stress de_L/deps for the twisted ECP geminal:
    complex kinetic stress + real Ewald(ee/ei/ii) + real ECP-local + complex ECP-nonlocal.
    Verified vs jax.grad(Re compute_local_energy) wrt strain, RT-fixed."""
    from jqmc.hamiltonians import compute_local_energy

    ham = _pat("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    sd = ham.structure_data
    eta = float(ed.eta)
    ri = np.asarray(ed.real_image_list, float)
    gv = np.asarray(ed.g_vectors, float)
    gf = np.asarray(ed.g_factors, float)
    cell = np.asarray(sd.cell, float)
    nmax = int(sd._n_max)
    z = np.asarray(cd._effective_charges, float)
    rion = np.asarray(ed.structure_data.positions, float)
    atol, rtol = get_tolerance_min(("wf_kinetic", "coulomb", "ewald", "swct", "local_energy"), "medium")
    for seed in (1, 2):
        ru, rd = _cfg(ham, seed)

        def fwd(eps):
            F = jnp.eye(3) + eps
            return jnp.real(compute_local_energy(ham._apply_strain(F), ru @ F.T, rd @ F.T, jnp.eye(3)))

        full = np.asarray(jax.grad(fwd)(jnp.zeros((3, 3))))
        total = (
            np.asarray(jqmc_rs.grad_T_wrt_strain_complex(gd, ru, rd))
            + np.asarray(jqmc_rs.grad_Eee_wrt_strain(ru, rd, eta, ri, gv, gf, cell, nmax))
            + np.asarray(jqmc_rs.grad_Eei_wrt_strain(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax))
            + np.asarray(jqmc_rs.grad_Eii_wrt_strain(eta, ri, gv, gf, z, rion, cell))
            + np.asarray(jqmc_rs.grad_V_ecp_local_wrt_strain(cd, ru, rd))
            + np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_complex(gd, cd, ru, rd, _W, _GP))
        )
        np.testing.assert_allclose(total, full, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


@pytest.mark.heavy
def test_grad_eL_R_complex_ECP_full():
    """FULL complex ECP local-energy FORCE de_L/dR for the twisted ECP geminal:
    complex kinetic force + Ewald(ei dR/ii) + ECP-local force + complex ECP-nonlocal
    force. Verified vs jax.grad(Re compute_local_energy) wrt atomic positions."""
    from jqmc.hamiltonians import compute_local_energy

    ham = _pat("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    pos = jnp.asarray(np.asarray(gd.orb_data_up_spin.aos_data.structure_data.positions))
    eta = float(ed.eta)
    cell = np.asarray(ed.structure_data.cell, float)
    nmax = int(gd.orb_data_up_spin.aos_data.structure_data._n_max)
    ri = np.asarray(ed.real_image_list, float)
    gv = np.asarray(ed.g_vectors, float)
    gf = np.asarray(ed.g_factors, float)
    z = np.asarray(cd._effective_charges, float)
    rion = np.asarray(ed.structure_data.positions, float)
    atol, rtol = get_tolerance_min(("wf_kinetic", "coulomb", "ewald", "local_energy"), "medium")
    for seed in (1, 2):
        ru, rd = _cfg(ham, seed)
        ref = np.asarray(
            jax.grad(
                lambda P: jnp.real(compute_local_energy(_inject_active_positions(ham, {"positions": P}), ru, rd, jnp.eye(3)))
            )(pos)
        )
        _, dR = jqmc_rs.grad_Eei(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax)
        total = (
            np.asarray(jqmc_rs.grad_T_wrt_R_complex(gd, ham.wavefunction_data.jastrow_data, ru, rd))
            + np.asarray(dR)
            + np.asarray(jqmc_rs.grad_Eii_wrt_R(eta, ri, gv, gf, z, rion, cell))
            + np.asarray(jqmc_rs.grad_eL_ecp_local_force_wrt_R(cd, ru, rd))
            + np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_R_complex(gd, cd, ru, rd, _W, _GP))
        )
        np.testing.assert_allclose(total, ref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


# --------------------------------------------------------------------------- #
# ECP non-local MO param-grad (jnone) (heavy)
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_grad_V_ecp_nonlocal_mo_complex():
    """Complex ECP non-local param gradient dV_nl/dMO (the optimization O_k of the
    ECP ratio) for the twisted ECP geminal. Verified vs jax.grad(ECP nonlocal energy)
    wrt complex MO."""
    ham = _pat("twist_sphepgto_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    cd = ham.coulomb_potential_data
    # The oracle evaluates the AOs on every ECP mesh point (det ratio), so the quantity
    # crosses ao_eval and det_ratio as well as coulomb/mo_grad (the XLA-Rust zone
    # resolver takes the zones actually crossed). The native
    # kernel is the MO gradient of the production nearest-neighbour ECP mesh
    # (NN_default nuclei per electron, as in `_dV_ecp_dc`), so the oracle uses the same
    # NN; NN=natom would add the far-nucleus V_l term the kernel deliberately omits.
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_ratio", "coulomb", "mo_grad"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        g = jax.grad(
            lambda G: compute_ecp_non_local_parts_nearest_neighbors(
                cd, wf.replace(geminal_data=G), ru, rd, jnp.eye(3), NN=NN_default, flag_determinant_only=True
            )[3]
        )(gd)
        gu, gdn = [np.asarray(x) for x in jqmc_rs.grad_V_ecp_nonlocal_wrt_mo_complex(gd, cd, ru, rd, _W, _GP)]
        np.testing.assert_allclose(
            gu, np.asarray(g.orb_data_up_spin.mo_coefficients), atol=atol, rtol=rtol, err_msg=f"mo_up seed={seed}"
        )
        np.testing.assert_allclose(
            gdn, np.asarray(g.orb_data_dn_spin.mo_coefficients), atol=atol, rtol=rtol, err_msg=f"mo_dn seed={seed}"
        )


# --------------------------------------------------------------------------- #
# withJ (J1/J2) kinetic value / MO-grad / force / stress (heavy)
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_kinetic_withj_complex():
    """Complex kinetic energy WITH Jastrow (J1/J2) for the twisted geminal. Verified vs
    jqmc compute_kinetic_energy on lithium_k025 j12 cplx."""
    ham = _pat("twist_sphepgto_j12")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    atol, rtol = _d8_backend_tolerance(("wf_kinetic", "jastrow_grad_lap", "mo_grad"), level="strict")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        tref = complex(compute_kinetic_energy(wf, ru, rd))
        re, im = jqmc_rs.compute_kinetic_withj_complex(gd, jd, ru, rd)
        np.testing.assert_allclose(complex(re, im), tref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


@pytest.mark.heavy
def test_grad_T_mo_withJ_complex():
    """Complex dT/dMO WITH Jastrow (J1/J2): exercises the gj!=0 path of
    complex_geminal_reverse. Verified vs jax.grad(Re kinetic_withJ) wrt complex MO on
    lithium_k025 j12 cplx."""
    ham = _pat("twist_sphepgto_j12")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    atol, rtol = _d8_backend_tolerance(("wf_kinetic", "jastrow_grad_lap", "mo_grad"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        g = jax.grad(lambda G: compute_kinetic_energy(wf.replace(geminal_data=G), ru, rd).real)(gd)
        gu, gdn = [np.asarray(x) for x in jqmc_rs.grad_T_mo_withJ_complex(gd, jd, ru, rd)]
        np.testing.assert_allclose(
            gu, np.asarray(g.orb_data_up_spin.mo_coefficients), atol=atol, rtol=rtol, err_msg=f"mo_up seed={seed}"
        )
        np.testing.assert_allclose(
            gdn, np.asarray(g.orb_data_dn_spin.mo_coefficients), atol=atol, rtol=rtol, err_msg=f"mo_dn seed={seed}"
        )


@pytest.mark.heavy
def test_grad_T_strain_withJ_complex():
    """Complex withJ kinetic STRESS dT/deps (twist + J1/J2). Oracle = a direct
    jax.jacrev(compute_kinetic_energy o production proper-strain), with the J2 cell strained."""
    ham = _pat("twist_sphepgto_j12")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    # kinetic-stress twist floor is the far-image AO cutoff (~2e-6), held below 1e-5.
    atol, rtol = _d8_backend_tolerance(("wf_kinetic", "swct", "jastrow_grad_lap", "mo_grad", "ao_eval"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        got = np.asarray(jqmc_rs.grad_T_wrt_strain_withJ_complex(gd, jd, ru, rd))
        ref = _kinetic_strain_withJ_oracle(ham, ru, rd)
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


@pytest.mark.heavy
def test_grad_T_withJ_J3_complex():
    """Complex Bloch-sphe J3 stress matches the direct XLA proper-strain oracle."""
    ham0 = _cell("lithium_k025.h5", True, (("j3", "pgto"),), "SD")
    gd = ham0.wavefunction_data.geminal_data
    jd = _randomize_j3(ham0.wavefunction_data.jastrow_data, seed=7)
    wf = ham0.wavefunction_data.replace(jastrow_data=jd)
    ham = ham0.replace(wavefunction_data=wf)
    # kinetic-stress twist floor is the far-image AO cutoff (~2e-6), held below 1e-5.
    atol_s, rtol_s = _d8_backend_tolerance(("wf_kinetic", "swct", "jastrow_grad_lap", "mo_grad", "ao_eval"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        st = np.asarray(jqmc_rs.grad_T_wrt_strain_withJ_complex(gd, jd, ru, rd))
        st_ref = _kinetic_strain_withJ_oracle(ham, ru, rd)
        np.testing.assert_allclose(st, st_ref, atol=atol_s, rtol=rtol_s, err_msg=f"stress seed={seed}")


@pytest.mark.heavy
def test_grad_T_withJ_J3_composite_complex():
    """Complex-twist COMPOSITE pgto(+)pw J3 kinetic FORCE + STRESS (twist + J1/J2/J3,
    randomized j_matrix): the J3 orbital is an MO-wrapped pgto(+)PW composite
    (``_J3_KIND['composite'] == 'mo'`` -> ``MOs_data(AOs_composite_data)``): the
    composite J3 force/stress case for the twisted (k != Gamma) system. The complex pGTO
    child contributes the atom-centered J3 kinetic force + value/lap/gou field strain; the
    complex plane-wave child (dAO/dR = 0) is zero force and strain-responsive only via |gk|
    (lap/gou), mirroring the real-Gamma composite J3. Force vs jax.grad(Re kinetic); stress
    vs the direct proper-strain jacrev oracle (J2 cell strained)."""
    ham = _cell("QEPROJ:pgto+pw:H_k025:AE", True, (("j3", "mo"),), "SD")
    gd = ham.wavefunction_data.geminal_data
    jd = _randomize_j3(ham.wavefunction_data.jastrow_data, seed=7)
    wf = ham.wavefunction_data.replace(jastrow_data=jd)
    ham = ham.replace(wavefunction_data=wf)
    pos = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.positions)
    # twist far-image AO cutoff floor (~2e-6) -> medium rtol, tight atol.
    atol, rtol = _d8_backend_tolerance(("wf_kinetic", "swct", "jastrow_grad_lap", "mo_grad", "ao_eval"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        st = np.asarray(jqmc_rs.grad_T_wrt_strain_withJ_complex(gd, jd, ru, rd))
        st_ref = _kinetic_strain_withJ_oracle(ham, ru, rd)
        np.testing.assert_allclose(st, st_ref, atol=atol, rtol=rtol, err_msg=f"stress seed={seed}")
        fc = np.asarray(jqmc_rs.grad_T_wrt_R_withJ_complex(gd, jd, ru, rd))
        fref = np.asarray(
            jax.grad(lambda P: compute_kinetic_energy(_inject_active_positions(wf, {"positions": P}), ru, rd).real)(pos)
        )
        np.testing.assert_allclose(fc, fref, atol=atol, rtol=rtol, err_msg=f"force seed={seed}")


# --------------------------------------------------------------------------- #
# withJ ECP non-local: energy / MO-grad / stress / force (heavy)
# 3-row loop: j12 + twist-j123ao(rand) + composite-j123ao(rand, Gamma k=0 cross-check).
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize("row", _WITHJ_ECP_ROWS)
def test_ecp_nonlocal_withJ_complex(row):
    """Complex ECP non-local energy WITH Jastrow: det ratio x exp(J(r')-J(r)),
    J=J1/J2 (Williamson) + J3 (Re(complex sphe AO) bilinear). vs jqmc
    compute_ecp_non_local_parts (flag_determinant_only=False)."""
    ham, gd, jd, cd, natom = _withj_ecp_row(row)
    wf = ham.wavefunction_data.replace(jastrow_data=jd)
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_ratio", "coulomb", "jastrow_ratio"), level="strict")
    for seed in (0, 1):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        re, im = jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, jd, cd, ru, rd, _W, _GP)
        ref = float(
            np.real(
                compute_ecp_non_local_parts_nearest_neighbors(
                    cd, wf, ru, rd, jnp.eye(3), NN=natom, flag_determinant_only=False
                )[3]
            )
        )
        np.testing.assert_allclose(re, ref, atol=atol, rtol=rtol, err_msg=f"{row} seed={seed}")


@pytest.mark.heavy
@pytest.mark.parametrize("row", _WITHJ_ECP_ROWS)
def test_grad_V_ecp_nonlocal_mo_withJ_complex(row):
    """Complex ECP non-local dV_nl/dMO WITH Jastrow (det MO param gradient; the Jastrow
    ratio exp(dJ) multiplies each mesh ratio, MO perp Jastrow). vs jax.grad of
    compute_ecp_non_local_parts(flag_determinant_only=False) wrt complex MO."""
    ham, gd, jd, cd, natom = _withj_ecp_row(row)
    wf = ham.wavefunction_data.replace(jastrow_data=jd)
    # The oracle evaluates the AOs on every ECP mesh point (det ratio and J3), so the
    # quantity crosses ao_eval as well (same zone set as test_ecp_nonlocal_withJ_complex).
    # The mixed
    # XLA oracle runs ao_eval at fp32, while Rust stays fp64 in both modes.
    # The native kernel is the MO gradient of the production nearest-neighbour ECP mesh
    # (NN_default nuclei per electron), so the oracle uses the same NN.
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_ratio", "coulomb", "jastrow_ratio", "mo_grad"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        gu, gdn = [np.asarray(x) for x in jqmc_rs.grad_V_ecp_nonlocal_wrt_mo_withJ_complex(gd, jd, cd, ru, rd, _W, _GP)]
        g = jax.grad(
            lambda G: (
                compute_ecp_non_local_parts_nearest_neighbors(
                    cd, wf.replace(geminal_data=G), ru, rd, jnp.eye(3), NN=NN_default, flag_determinant_only=False
                )[3].real
            )
        )(gd)
        ru_ref = np.asarray(g.orb_data_up_spin.mo_coefficients)
        rd_ref = np.asarray(g.orb_data_dn_spin.mo_coefficients)
        np.testing.assert_allclose(gu, ru_ref, atol=atol, rtol=rtol, err_msg=f"{row} mo_up seed={seed}")
        np.testing.assert_allclose(gdn, rd_ref, atol=atol, rtol=rtol, err_msg=f"{row} mo_dn seed={seed}")


@pytest.mark.heavy
@pytest.mark.parametrize("row", _WITHJ_ECP_ROWS)
def test_grad_V_ecp_nonlocal_strain_withJ_complex(row):
    """Complex ECP non-local STRESS dV_nl/deps WITH Jastrow (J1/J2 Williamson + J3 =
    Re(complex sphe AO)). Oracle = jax.grad of the ECP non-local energy wrt strain using
    the production full-wf deform (`ham._apply_strain`, with the J2 cell strained)."""
    ham, gd, jd, cd, natom = _withj_ecp_row(row)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    atol, rtol = _d8_backend_tolerance(("coulomb", "swct", "jastrow_grad_lap"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)

        def fwd(eps):
            F = jnp.eye(3) + eps
            ham2 = ham._apply_strain(F)
            return compute_ecp_non_local_parts_nearest_neighbors(
                ham2.coulomb_potential_data,
                ham2.wavefunction_data,
                ru @ F.T,
                rd @ F.T,
                jnp.eye(3),
                NN=natom,
                flag_determinant_only=False,
            )[3].real

        got = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_withJ_complex(gd, jd, cd, ru, rd, _W, _GP))
        ref = np.asarray(jax.grad(fwd)(jnp.zeros((3, 3))))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{row} seed={seed}")


@pytest.mark.heavy
@pytest.mark.parametrize("row", _WITHJ_ECP_ROWS)
def test_grad_V_ecp_nonlocal_R_withJ_complex(row):
    """Complex ECP non-local FORCE dV_nl/dR WITH Jastrow (J1 Williamson value-force +
    complex J3 value-force; J2 e-e has no nuclear force). Oracle = jax.grad of the ECP
    non-local energy wrt atomic positions (full-wf _inject_active_positions)."""
    ham, gd, jd, cd, natom = _withj_ecp_row(row)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    pos = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.positions)
    atol, rtol = _d8_backend_tolerance(("coulomb", "jastrow_grad_lap"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)

        def fwd(P):
            ham2 = _inject_active_positions(ham, {"positions": P})
            return compute_ecp_non_local_parts_nearest_neighbors(
                ham2.coulomb_potential_data, ham2.wavefunction_data, ru, rd, jnp.eye(3), NN=natom, flag_determinant_only=False
            )[3].real

        got = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_R_withJ_complex(gd, jd, cd, ru, rd, _W, _GP))
        ref = np.asarray(jax.grad(fwd)(jnp.asarray(pos)))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{row} seed={seed}")


# --------------------------------------------------------------------------- #
# DTN-in-twist (heavy)
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_ecp_nonlocal_withDTN_complex():
    """DTN-in-twist: complex ECP non-local energy with a DTN Jastrow. DTN is a real
    distance-based Jastrow, so it enters the complex ratio as exp(dJ_DTN) unchanged
    (parallel to J1/J2/J3). vs jqmc nearest_neighbors(flag_determinant_only=False).

    Only the k025 cell in-suite: the jqmc jitted nn_ecp reference throws a pytree
    metadata error when replayed across DIFFERENT-cell fixtures in one xdist worker."""
    ham = _pat("twist_sphepgto_jdtn")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cd = ham.coulomb_potential_data
    natom = np.asarray(ham.structure_data.positions).shape[0]
    atol, rtol = _d8_backend_tolerance(("ao_eval", "det_ratio", "coulomb", "jastrow_ratio"), level="strict")
    for seed in (0, 1):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        ref = float(
            np.real(
                compute_ecp_non_local_parts_nearest_neighbors(
                    cd, wf, ru, rd, jnp.eye(3), NN=natom, flag_determinant_only=False
                )[3]
            )
        )
        re, im = jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, jd, cd, ru, rd, _W, _GP)
        np.testing.assert_allclose(re, ref, atol=atol, rtol=rtol, err_msg=f"seed={seed}")


@pytest.mark.heavy
def test_grad_V_ecp_nonlocal_withDTN_strain_force_complex():
    """DTN-in-twist STRESS dV_nl/deps and FORCE dV_nl/dR with a DTN Jastrow. DTN enters
    the value-ratio (exp dJ_DTN), the strain (dtn_value_strain, min-image) and the
    nuclear force (grad_j_dtn_wrt_r_atom). Oracle = jax.grad of the full-wf deform."""
    ham = _pat("twist_sphepgto_jdtn")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    natom = np.asarray(ham.structure_data.positions).shape[0]
    pos = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.positions)
    atol_s, rtol_s = _d8_backend_tolerance(("coulomb", "swct", "jastrow_grad_lap"), level="medium")
    atol_f, rtol_f = _d8_backend_tolerance(("coulomb", "jastrow_grad_lap"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)

        def fwd_s(eps):
            F = jnp.eye(3) + eps
            ham2 = ham._apply_strain(F)
            return compute_ecp_non_local_parts_nearest_neighbors(
                ham2.coulomb_potential_data,
                ham2.wavefunction_data,
                ru @ F.T,
                rd @ F.T,
                jnp.eye(3),
                NN=natom,
                flag_determinant_only=False,
            )[3].real

        ref_s = np.asarray(jax.grad(fwd_s)(jnp.zeros((3, 3))))
        got_s = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_withJ_complex(gd, jd, cd, ru, rd, _W, _GP))
        np.testing.assert_allclose(got_s, ref_s, atol=atol_s, rtol=rtol_s, err_msg=f"stress seed={seed}")

        def fwd_f(P):
            ham2 = _inject_active_positions(ham, {"positions": P})
            return compute_ecp_non_local_parts_nearest_neighbors(
                ham2.coulomb_potential_data, ham2.wavefunction_data, ru, rd, jnp.eye(3), NN=natom, flag_determinant_only=False
            )[3].real

        ref_f = np.asarray(jax.grad(fwd_f)(jnp.asarray(pos)))
        got_f = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_R_withJ_complex(gd, jd, cd, ru, rd, _W, _GP))
        np.testing.assert_allclose(got_f, ref_f, atol=atol_f, rtol=rtol_f, err_msg=f"force seed={seed}")


@pytest.mark.heavy
def test_complex_kinetic_DTN_stress_force():
    """Complex-twist DTN (u+chi+f) KINETIC stress dT/deps + force dT/dR. The DTN is a
    real distance-based Jastrow, so its kinetic term enters via dtn_kstress / force added
    to the complex det kinetic. Oracle = jax.grad(compute_kinetic_energy) with the DTN
    structure strained / injected. f-channel active (randomize_dtn keeps use_f)."""
    ham0 = _pat("twist_sphepgto_jdtn")
    jd = _randomize_dtn(ham0.wavefunction_data.jastrow_data, seed=5)
    assert bool(jd.jastrow_dtn_data.use_f)
    wf = ham0.wavefunction_data.replace(jastrow_data=jd)
    # NB: the committed jdtn builder ships J1+J2+DTN together (with_jastrow=True), so the
    # The kinetic-stress oracle strains J1/J2/J3/DTN with the physical lattice,
    # exactly as `ham._apply_strain` specifies.
    ham = ham0.replace(wavefunction_data=wf)
    gd = wf.geminal_data
    P0 = jnp.asarray(np.asarray(ham.structure_data.positions, float))
    atol_s, rtol_s = _d8_backend_tolerance(("wf_kinetic", "swct", "jastrow_grad_lap"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)

        st = np.asarray(jqmc_rs.grad_T_wrt_strain_withJ_complex(gd, jd, ru, rd))
        st_ref = _kinetic_strain_withJ_oracle(ham, ru, rd)
        np.testing.assert_allclose(st, st_ref, atol=atol_s, rtol=rtol_s, err_msg=f"stress seed={seed}")

        def Tf(P):
            return jnp.real(compute_kinetic_energy(_inject_active_positions(wf, {"positions": P}), ru, rd))

        fc = np.asarray(jqmc_rs.grad_T_wrt_R_withJ_complex(gd, jd, ru, rd))
        fc_ref = np.asarray(jax.grad(Tf)(P0))
        np.testing.assert_allclose(fc, fc_ref, atol=atol_s, rtol=rtol_s, err_msg=f"force seed={seed}")


# --------------------------------------------------------------------------- #
# Complex-twist COMPOSITE (pgto+pw) kinetic stress + force with J3 (heavy)
# The complex kernel is fed a REAL-Gamma pgto(+)pw child: a legitimate k = 0
# cross-check (complex kernel reduces to the real result at k = 0), HARD-asserted.
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_complex_composite_kinetic_stress_force():
    """Complex-twist COMPOSITE (pgto+pw) determinant kinetic STRESS + FORCE with J3.
    Oracle: direct jax.jacrev(compute_kinetic_energy o proper-strain) for the stress
    (g6w replacement); jax.grad(compute_kinetic_energy) for the force. The complex kernel
    accepts the REAL-Gamma plane-wave child (k = 0 degeneracy, gk == G) and reduces to the
    real result, so this is HARD-asserted against the jqmc oracle."""
    ham0 = _cell("QE:Li_gamma_pgto+pw", True, (("j3", "mo"),), "SD")
    jd = _randomize_j3(ham0.wavefunction_data.jastrow_data, seed=7)
    ham = ham0.replace(wavefunction_data=ham0.wavefunction_data.replace(jastrow_data=jd))
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    atol, rtol = _d8_backend_tolerance(("wf_kinetic", "swct", "jastrow_grad_lap", "mo_grad"), level="medium")
    for seed in (1, 2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, seed)
        # kinetic stress vs the direct jax.jacrev oracle (g6w replacement)
        st = np.asarray(jqmc_rs.grad_T_wrt_strain_withJ_complex(gd, jd, ru, rd))
        st_ref = _kinetic_strain_withJ_oracle(ham, ru, rd)
        np.testing.assert_allclose(st, st_ref, atol=atol, rtol=rtol, err_msg=f"stress seed={seed}")
        # kinetic force vs jax.grad(compute_kinetic_energy o inject R)
        P0 = jnp.asarray(np.asarray(ham.structure_data.positions, float))

        def Tf(P):
            return jnp.real(compute_kinetic_energy(_inject_active_positions(wf, {"positions": P}), ru, rd))

        fc = np.asarray(jqmc_rs.grad_T_wrt_R_withJ_complex(gd, jd, ru, rd))
        fc_ref = np.asarray(jax.grad(Tf)(P0))
        np.testing.assert_allclose(fc, fc_ref, atol=atol, rtol=rtol, err_msg=f"force seed={seed}")

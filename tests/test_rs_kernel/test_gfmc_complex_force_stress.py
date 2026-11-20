"""Rust CPU-kernel equivalence: complex-twist (k != 0) GFMC/LRDMC HF force + stress.

Asserts the Rust analytic kernels

    jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex   (force, returns (dR, dr_up, dr_dn))
    jqmc_rs._gfmc_stress_ecp_pbc_complex     (stress, returns (3, 3))

reproduce jqmc's XLA analytic GFMC force/stress oracle
``jqmc.derivatives.gfmc.grad_e_L_{t,n}_wrt_{positions,strain}`` on the COMMITTED
complex-twist ECP fixture ``tests/trexio_example_files/lithium_k025.h5``
(NO external bench h5).

At a twist the wave-function and ECP derivative intermediates are complex
(holomorphic in the complex Bloch orbitals); the observable HF force/stress is
real, so the Rust kernel projects to ``Re`` only at the boundary and the oracle
is compared component-by-component. GFMC_t and GFMC_n delegate to the same
novar=0 continuum-measurement derivative and are both checked here.

The seed/alat scan is an adversarial near-node
configuration and independently keeps propagation S-prime membership live.
S-prime is not consumed by the measurement derivative. jnone / j12 (J1+J2) /
j123ao (AO-leaf J3) are covered.
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
from jqmc._setting import ATOL_RS_GFMC_SPRIME_FAST_VS_FULL, RTOL_RS_GFMC_SPRIME_FAST_VS_FULL
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import gfmc
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

from conftest import trexio_path  # test_rs_kernel is not a package; pytest prepends its dir to sys.path
from tests._ecp_core_electrons import core_electrons_of

# Mandatory CWD classification (tests/conftest.py): these tests read only ABSOLUTE committed-fixture
# paths (``trexio_path``), so the autouse ``_isolate_cwd`` isolation is harmless -- ``with_chdir``.
pytestmark = [pytest.mark.with_chdir]

_FIXTURE = "lithium_k025.h5"
# jnone + j12 (J1+J2) + DTN (u/chi/f) + j123 (J1+J2+J3, complex-sphe Bloch J3) are the covered WF
# variants. The sampled coordinates also retain non-empty propagation S-prime
# membership for the independent propagation oracle; the measurement derivative
# itself is continuum-only.
_KINDS = ["jnone", "j12", "dtn", "j123"]

# ECP non-local move estimator.  ``dltmove`` (DLA) forms a DIFFERENT e_L from ``tmove``: the
# fixed-node-clamped hop uses the DETERMINANT-only amplitude and re-weights only the accepted (FN)
# part by the Jastrow ratio, so BOTH the force and the stress carry a det-only strain-Pulay /
# ``use_full`` branch.  Covering only ``tmove`` would leave a complex ``dltmove`` run silently
# returning the t-move derivative for the DLA Hamiltonian,
# so both estimators are compared against the XLA oracle, which takes ``non_local_move`` directly.
# ``jnone`` is deliberately kept in the matrix even though ``jr = 1`` makes dltmove == tmove there:
# it pins that the det-only branch does NOT perturb the no-Jastrow answer.
_NLM = ["tmove", "dltmove"]

# Match the global 12-point production quadrature used by the XLA GFMC oracle.
# Nearest-neighbor truncation remains NN=1.
_WEIGHTS = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_GRID = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
_NN = 1


def _sym_j3(jd):
    """Attach a SYMMETRIC random J3 coupling block to a J3-carrying jastrow_data.  The jqmc J3
    kernels deliberately EXPLOIT ``j3_mat.T @ x == j3_mat @ x``, and
    ``Jastrow_three_body_data.sanity_check`` RAISES on a non-symmetric coupling block, so a random
    j_matrix MUST be symmetrized (``jm[:, :-1] = 0.5*(jm[:, :-1] + jm[:, :-1].T)``); the last column
    is the one-body linear term and is left alone.
    """
    import jax.numpy as jnp

    j3 = jd.jastrow_three_body_data
    rng = np.random.default_rng(7)
    jm = 0.05 * rng.standard_normal(np.asarray(j3.j_matrix).shape)
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    return jd.replace(jastrow_three_body_data=j3.replace(j_matrix=jnp.asarray(jm)))


def _build_ham(kind: str) -> Hamiltonian_data:
    if kind == "j123":
        # Active complex-sphe Bloch J3 (AO-leaf).  The named ``twist_sphepgto_j123`` pattern
        # (``_rs_fixtures``) attaches J1+J2 as well; this row needs J3 ONLY, so build the
        # twist+J3 ham via the shared ``_build_ham_for_cell`` (imported read-only), then swap in
        # a SYMMETRIC random j_matrix (the shipped one may be zero -> a vacuous J3).
        # Built J3-ONLY (with_jastrow=False): an exp J1+J2 smooths the
        # hopping enough that propagation S-prime membership need not fire (the
        # reason ``j12`` is exempted from the independent membership assertion
        # below). J3-alone keeps the adversarial configuration.
        from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

        ham = _build_ham_for_cell(_FIXTURE, False, {"j3": "pgto"}, "SD")
        jd = _sym_j3(ham.wavefunction_data.jastrow_data)
        return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    (sd, aos_data, _mu, _md, gem, cpd) = read_trexio_file(trexio_path(_FIXTURE), store_tuple=True)
    if kind == "jnone":
        jd = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)
    elif kind == "dtn":
        r_ins = sd._wigner_seitz_inscribed_radius()
        dtn = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),  # Li ccECP: Z=3, two core electrons
            pseudized_per_species=(True,),
            use_u=True,
            use_chi=True,
            use_f=True,
            cutoff_smoothness=4,
            n_order_u=4,
            n_order_chi=4,
            n_order_eN=2,
            n_order_ee=2,
            cutoff_ratio_u=3.0 / r_ins,
            cutoff_ratio_chi=3.0 / r_ins,
            # Resolve from the cell geometry: for this fixture R_inscribed is
            # 3.316 bohr, so a fixed 3.0 would put the min-image seam inside
            # the f-term support (the smoothness contract is 2*L_f < R_ins).
            cutoff_ratio_f=0.75,
            random_init=True,
            seed=3,
        )
        jd = Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
        )
    else:  # j12 = J1(exp) + J2(exp)
        n_atom = len(sd.positions)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=sd)
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0, structure_data=sd, core_electrons=core_electrons_of(cpd, n_atom), jastrow_1b_type="exp"
        )
        jd = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=gem)
    return Hamiltonian_data(structure_data=sd, coulomb_potential_data=cpd, wavefunction_data=wf)


def _rot(seed: int):
    import jax.numpy as jnp

    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return jnp.asarray(q)


def _twist_config(ham):
    """Find an adversarial twist configuration with live propagation membership.

    The tight sampling keeps electrons close to the nuclei so the Rust pGTO
    far-image AO cutoff remains below the equivalence tolerance. The returned
    S-prime flag is propagation-only and is not consumed by the novar=0
    measurement derivative. Falls back to the last scanned configuration.
    """
    import jax.numpy as jnp

    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    pos = np.asarray(ham.structure_data.positions)
    last = None
    for seed in range(12):
        rng = np.random.default_rng(seed)
        r_up = jnp.asarray(pos[:nu] + rng.normal(scale=0.5, size=(nu, 3)))
        r_dn = jnp.asarray(pos[:nd] + rng.normal(scale=0.5, size=(nd, 3)))
        for alat in (1.5, 2.0, 2.5):
            RT = _rot(seed + 2)
            w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
            last = (r_up, r_dn, RT, alat)
            if float(np.sum(np.asarray(w))) > 0.0:
                return r_up, r_dn, RT, alat, True
    return (*last, False)


def _rt_np(RT):
    return np.asarray(RT, dtype=np.float64)


_ZONES_JNONE = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
_ZONES_WITH_JASTROW = _ZONES_JNONE + ("jastrow_eval", "jastrow_grad_lap")


# One explicit zone list per ``kind`` parameter (never their union). The table is a
# literal so each case's list is read statically by its per-case executable check.
_ZONES_BY_KIND = {
    "jnone": _ZONES_JNONE,
    "j12": _ZONES_WITH_JASTROW,
    "dtn": _ZONES_WITH_JASTROW,
    "j123": _ZONES_WITH_JASTROW,
}


@pytest.mark.heavy
@pytest.mark.parametrize("nlm", _NLM)
@pytest.mark.parametrize("kind", _KINDS)
def test_gfmc_complex_force(kind, nlm):
    """GFMC complex force."""
    ham = _build_ham(kind)
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    if kind != "j12":
        assert nonempty, f"expected live propagation S-prime membership for adversarial {kind} fixture"
    wf = ham.wavefunction_data
    ru = np.asarray(r_up, dtype=np.float64)
    rd = np.asarray(r_dn, dtype=np.float64)

    dR, dru, drd = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        ru,
        rd,
        alat,
        _rt_np(RT),
        _WEIGHTS,
        _GRID,
        _NN,
        nlm == "dltmove",
    )
    # The compatibility ``debug`` surface delegates to the same novar=0
    # continuous measurement kernel. Keep this assertion to pin the public
    # production/debug contract while XLA parity below supplies an independent
    # numerical oracle.
    dR_debug, dru_debug, drd_debug = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        ru,
        rd,
        alat,
        _rt_np(RT),
        _WEIGHTS,
        _GRID,
        _NN,
        nlm == "dltmove",
        debug=True,
    )
    for label, fast, full in (
        ("dR", dR, dR_debug),
        ("dr_up", dru, dru_debug),
        ("dr_dn", drd, drd_debug),
    ):
        np.testing.assert_allclose(
            np.asarray(fast),
            np.asarray(full),
            atol=ATOL_RS_GFMC_SPRIME_FAST_VS_FULL,
            rtol=RTOL_RS_GFMC_SPRIME_FAST_VS_FULL,
            err_msg=f"Rust production/debug measurement-force mismatch ({label}, {kind}, {nlm})",
        )

    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_BY_KIND[kind], level="medium")
    for estimator, fn in (("t", gfmc.grad_e_L_t_wrt_positions), ("n", gfmc.grad_e_L_n_wrt_positions)):
        oR, oru, ord_ = fn(ham, r_up, r_dn, RT, alat, non_local_move=nlm)
        np.testing.assert_allclose(
            np.asarray(dR),
            np.real(np.asarray(oR)),
            atol=atol,
            rtol=rtol,
            err_msg=f"dR mismatch ({kind}, {nlm}, GFMC_{estimator})",
        )
        np.testing.assert_allclose(
            np.asarray(dru),
            np.real(np.asarray(oru)),
            atol=atol,
            rtol=rtol,
            err_msg=f"dr_up mismatch ({kind}, {nlm}, GFMC_{estimator})",
        )
        np.testing.assert_allclose(
            np.asarray(drd),
            np.real(np.asarray(ord_)),
            atol=atol,
            rtol=rtol,
            err_msg=f"dr_dn mismatch ({kind}, {nlm}, GFMC_{estimator})",
        )


@pytest.mark.heavy
@pytest.mark.parametrize("nlm", _NLM)
@pytest.mark.parametrize("kind", _KINDS)
def test_gfmc_complex_stress(kind, nlm):
    """GFMC complex stress."""
    ham = _build_ham(kind)
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    if kind != "j12":
        assert nonempty, f"expected live propagation S-prime membership for adversarial {kind} fixture"
    wf = ham.wavefunction_data
    ru = np.asarray(r_up, dtype=np.float64)
    rd = np.asarray(r_dn, dtype=np.float64)

    sigma = jqmc_rs._gfmc_stress_ecp_pbc_complex(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        ru,
        rd,
        alat,
        _rt_np(RT),
        _WEIGHTS,
        _GRID,
        _NN,
        nlm == "dltmove",
    )
    sigma_debug = jqmc_rs._gfmc_stress_ecp_pbc_complex(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        ru,
        rd,
        alat,
        _rt_np(RT),
        _WEIGHTS,
        _GRID,
        _NN,
        nlm == "dltmove",
        debug=True,
    )
    np.testing.assert_allclose(
        np.asarray(sigma),
        np.asarray(sigma_debug),
        atol=ATOL_RS_GFMC_SPRIME_FAST_VS_FULL,
        rtol=RTOL_RS_GFMC_SPRIME_FAST_VS_FULL,
        err_msg=f"Rust production/debug measurement-stress mismatch ({kind}, {nlm})",
    )

    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_BY_KIND[kind], level="medium")
    for estimator, fn in (("t", gfmc.grad_e_L_t_wrt_strain), ("n", gfmc.grad_e_L_n_wrt_strain)):
        osig = fn(ham, r_up, r_dn, RT, alat, non_local_move=nlm)
        np.testing.assert_allclose(
            np.asarray(sigma),
            np.real(np.asarray(osig)),
            atol=atol,
            rtol=rtol,
            err_msg=f"stress mismatch ({kind}, {nlm}, GFMC_{estimator})",
        )


@pytest.mark.heavy
@pytest.mark.parametrize("nlm", _NLM)
def test_gfmc_zero_k_complex_bspline_dtn_force_stress(nlm, tmp_path):
    """Blocking derivative parity for v3 Gamma-as-complex B-spline inputs.

    Nonzero-twist pGTO coverage above cannot prove that a zero-k complex pure
    B-spline or pGTO+B-spline composite survives the production converter and
    loader.  Both AO forms are kept in one pytest item so xdist cannot conceal
    cross-kind JIT metadata contamination.  Force and stress are compared to
    both GFMC_t and GFMC_n XLA analytical implementations; no skip, xfail,
    allowlist, or JAX fallback is permitted.
    """
    from tests.test_rs_kernel._rs_fixtures import build_qe_zero_k_complex_dtn_ham

    for ao_kind in ("bspline", "pgto+bspline"):
        tag = f"zero_k_complex_{ao_kind.replace('+', '_')}_dtn_{nlm}"
        ham = build_qe_zero_k_complex_dtn_ham(ao_kind, tmp_path / f"{tag}.h5")
        r_up, r_dn, RT, alat, _nonempty = _twist_config(ham)
        wf = ham.wavefunction_data
        ru = np.asarray(r_up, dtype=np.float64)
        rd = np.asarray(r_dn, dtype=np.float64)

        dR, dru, drd = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            ru,
            rd,
            alat,
            _rt_np(RT),
            _WEIGHTS,
            _GRID,
            _NN,
            nlm == "dltmove",
        )
        sigma = jqmc_rs._gfmc_stress_ecp_pbc_complex(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            ru,
            rd,
            alat,
            _rt_np(RT),
            _WEIGHTS,
            _GRID,
            _NN,
            nlm == "dltmove",
        )

        atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_WITH_JASTROW, level="medium")
        for estimator, force_fn, stress_fn in (
            ("t", gfmc.grad_e_L_t_wrt_positions, gfmc.grad_e_L_t_wrt_strain),
            ("n", gfmc.grad_e_L_n_wrt_positions, gfmc.grad_e_L_n_wrt_strain),
        ):
            oR, oru, ord_ = force_fn(ham, r_up, r_dn, RT, alat, non_local_move=nlm)
            osig = stress_fn(ham, r_up, r_dn, RT, alat, non_local_move=nlm)
            for name, got, ref in (
                ("dR", dR, oR),
                ("dr_up", dru, oru),
                ("dr_dn", drd, ord_),
                ("stress", sigma, osig),
            ):
                np.testing.assert_allclose(
                    np.asarray(got),
                    np.real(np.asarray(ref)),
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"{name} mismatch ({tag}, GFMC_{estimator})",
                )

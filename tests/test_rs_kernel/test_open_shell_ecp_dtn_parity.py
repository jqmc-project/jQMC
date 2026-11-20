"""Native vs XLA parity: open-shell (N_up != N_dn) ccECP wavefunctions with an active DTN Jastrow.

The native one-electron-move ECP mesh kernels of the DTN Jastrow address a moved
down-spin electron by its global stacked index ``n_up + li``:
``param_grad_dtn::DtnMovedOk::moved_grads`` (DTN u/chi/f dE_L/dc, ECP term),
``forces_dtn::grad_j_dtn_wrt_r_atom_move_delta`` (DTN dJ/dR Pulay delta of every ECP
mesh candidate of the HF force) and ``stress_pbc::dtn_value_strain_move_delta_batch``
(DTN value-strain delta of every ECP mesh candidate of the periodic HF stress). Every
other ECP derivative parity fixture with a DTN is closed-shell (water 4/4, lithium
1/1, Cl2 7/7), where ``n_up + li == n_dn + li`` and a wrong spin offset is invisible
(issue #798).

The inputs are

  * molecular: ``water_ccecp_ccpvqz.h5`` SD + J1 + J2 + DTN (u/chi/f, randomized
    coefficients), with the down-spin block cut to 3 molecular orbitals with
    independent coefficients and ``time_reversal=False`` (4 up / 3 dn, the recipe of
    ``test_ecp_unrestricted_wdot.py``);
  * periodic: the committed QE ``O-lsda_gamma`` save (real Gamma, triplet oxygen,
    ccECP, 4 up / 2 dn, independent up/dn orbitals) converted by ``run_qe2jqmc``
    (kind=pgto), with the same DTN channels attached.

The checks are native vs XLA at fixed configurations:

  (a) DTN u/chi/f ECP parameter derivatives, native ``grad_eL_ecp_dtn_params`` vs
      ``jax.grad`` of the XLA non-local ECP energy wrt the DTN coefficients;
  (b) molecular HF force dE_L/d(R, r_up, r_dn), native ``vmc_grad_e_L_ecp`` vs
      ``assemble.compute_grad_e_L``, with every electron drawn close to the oxygen
      ECP centre so that down-spin electrons carry live non-local candidates;
  (c) periodic HF stress dE_L/d(eps), native ``vmc_grad_e_L_eps_pbc`` vs
      ``assemble.compute_grad_e_L(wrt="eps")``.

Tolerances: (a) uses the strict Rust/XLA tier. The HF force and stress aggregates (b)
and (c) use the medium tier, which equals strict in full precision; in mixed precision
the strict fp32 pair (2e-7, 2e-6) is below the fp32 cancellation of these O(10^2)
aggregates on the clean code (force dR: max |diff| 1.6e-4 at |dR| = 33, stress: 3.8e-5
at |sigma| = 13, i.e. 5e-6 to 9e-6 relative per element), the same reason every other
force/stress parity row (``test_vmc_forces.py``, ``test_periodic_lsda.py``) uses medium.
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

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors  # noqa: E402
from jqmc.derivatives import assemble  # noqa: E402
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from jqmc.jastrow_factor import Jastrow_DTN_data  # noqa: E402
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched  # noqa: E402
from jqmc.wavefunction import Wavefunction_data  # noqa: E402
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg  # noqa: E402
from tests.test_rs_kernel.test_param_grad_el import _assert_dtn_liveness, _randomize_dtn  # noqa: E402
from tests.test_rs_kernel.test_periodic_lsda import _build_gamma  # noqa: E402

pytestmark = [pytest.mark.with_chdir]

_N_UP = 4
_N_DN_MOL = 3
_N_DN_PBC = 2
_ZONES_ECP_PARAM = ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval")
_ZONES_FORCE = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)


def _with_jastrow(ham, jd):
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


@pytest.fixture(scope="module")
def mol_ham():
    """Water ccECP SD + J1 + J2 + active DTN, 4 up / 3 dn with independent down orbitals."""
    ham = _build_ham_for_cell("water_ccecp_ccpvqz.h5", True, {"dtn": True}, "SD")
    ham = _with_jastrow(ham, _randomize_dtn(ham))
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    down = gd.orb_data_dn_spin
    c_down = np.asarray(down.mo_coefficients[:_N_DN_MOL])
    rng = np.random.default_rng(5)
    independent = c_down + 0.15 * np.max(np.abs(c_down)) * rng.standard_normal(c_down.shape)
    gd = gd.replace(
        num_electron_dn=_N_DN_MOL,
        orb_data_dn_spin=down.replace(num_mo=_N_DN_MOL, mo_coefficients=independent),
        time_reversal=False,
    )
    ham = ham.replace(wavefunction_data=wf.replace(geminal_data=gd))
    ham.sanity_check()
    assert (int(gd.num_electron_up), int(gd.num_electron_dn)) == (_N_UP, _N_DN_MOL)
    assert ham.coulomb_potential_data.ecp_flag
    d = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    assert d.use_u and d.use_chi and d.use_f
    assert all(np.any(np.asarray(getattr(d, k)) != 0) for k in ("alpha_indep", "beta_indep", "gamma_indep"))
    return ham


@pytest.fixture(scope="module")
def pbc_ham(tmp_path_factory):
    """QE O-lsda_gamma (pgto, ccECP, 4 up / 2 dn) + J1/J2 + active DTN u/chi/f."""
    ham = _build_gamma("pgto", tmp_path_factory.mktemp("o_lsda_gamma"))
    structure, coulomb = ham.structure_data, ham.coulomb_potential_data
    numbers = np.asarray(structure.atomic_numbers)
    effective = np.asarray(coulomb._effective_charges)
    species = np.unique(numbers)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=tuple(float(np.mean(effective[numbers == z])) for z in species),
        pseudized_per_species=tuple(bool(np.all(coulomb.pseudized_per_atom[numbers == z])) for z in species),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=8,
        n_order_chi=8,
        n_order_eN=4,
        n_order_ee=4,
        random_init=True,
        seed=13,
    )
    ham = _with_jastrow(ham, ham.wavefunction_data.jastrow_data.replace(jastrow_dtn_data=dtn))
    ham = _with_jastrow(ham, _randomize_dtn(ham))
    ham.sanity_check()
    gd = ham.wavefunction_data.geminal_data
    assert (int(gd.num_electron_up), int(gd.num_electron_dn)) == (_N_UP, _N_DN_PBC)
    assert ham.coulomb_potential_data.ecp_flag and bool(ham.structure_data.pbc_flag)
    return ham


def _mol_geminal_inv(ham, r_up, r_dn):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])
    # The native aggregate keeps the running inverse in fp64 regardless of the AO zone dtype.
    return np.asarray(ginv[0], dtype=np.float64)


def _near_ecp_centre_cfg(ham, seed, scale):
    """Every electron Gaussian-distributed around the oxygen ECP centre (largest atomic number)."""
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    i_o = int(np.argmax(np.asarray(ham.structure_data.atomic_numbers)))
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    r_up = R[i_o] + rng.normal(scale=scale, size=(int(gd.num_electron_up), 3))
    r_dn = R[i_o] + rng.normal(scale=scale, size=(int(gd.num_electron_dn), 3))
    return r_up, r_dn


def _down_spin_nonlocal_weight(ham, r_up, r_dn):
    """Largest |V_nl| of the XLA non-local candidates that move a down-spin electron."""
    _, mesh_dn, v_nl, _ = compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data, ham.wavefunction_data, r_up, r_dn, np.eye(3)
    )
    mesh_dn = np.asarray(mesh_dn).reshape(-1, len(r_dn), 3)
    moved_dn = np.any(mesh_dn != np.asarray(r_dn)[None], axis=(1, 2))
    v_nl = np.asarray(v_nl).reshape(-1)
    assert moved_dn.shape == v_nl.shape and np.any(moved_dn)
    return float(np.max(np.abs(v_nl[moved_dn])))


@pytest.mark.light
@pytest.mark.parametrize("seed", [1])
def test_ecp_dtn_param_grads_open_shell(mol_ham, seed):
    """(a) DTN u/chi/f dE_L/dc (ECP non-local term), native == XLA autodiff, 4 up / 3 dn."""
    ham = mol_ham
    jd = ham.wavefunction_data.jastrow_data
    d = jd.jastrow_dtn_data
    cd = ham.coulomb_potential_data
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=seed, scale=1.0, cond_max=100.0)
    RT = np.eye(3)
    alpha, beta, gamma = jqmc_rs.grad_eL_ecp_dtn_params(gem, jd, cd, r_up, r_dn, _W, _GP)

    def ref(field, value):
        def fwd(x):
            jd2 = jd.replace(jastrow_dtn_data=d.replace(**{field: x}))
            return compute_ecp_non_local_parts_nearest_neighbors(
                cd, Wavefunction_data(jastrow_data=jd2, geminal_data=gem), r_up, r_dn, RT
            )[3]

        return np.asarray(jax.grad(fwd)(np.asarray(value)))

    _assert_dtn_liveness(d)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_PARAM, level="strict")
    for tag, got, field, value in (
        ("u", alpha, "alpha_indep", d.alpha_indep),
        ("chi", beta, "beta_indep", d.beta_indep),
        ("f", gamma, "gamma_indep", d.gamma_indep),
    ):
        expected = ref(field, value)
        assert expected.size and np.max(np.abs(expected)) > 0.0, tag
        np.testing.assert_allclose(
            np.asarray(got), expected, atol=atol, rtol=rtol, err_msg=f"seed={seed} DTN {tag} ECP dE_L/dc"
        )


@pytest.mark.light
@pytest.mark.parametrize("seed", [3])
def test_ecp_dtn_hf_force_open_shell(mol_ham, seed):
    """(b) Molecular HF force dE_L/d(R, r_up, r_dn), native == XLA, 4 up / 3 dn, electrons near the ECP centre."""
    ham = mol_ham
    wf = ham.wavefunction_data
    r_up, r_dn = _near_ecp_centre_cfg(ham, seed, 0.6)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_FORCE, level="medium")
    # Fixture precondition: a down-spin electron has live non-local ECP candidates.
    assert _down_spin_nonlocal_weight(ham, r_up, r_dn) > 1.0e3 * atol
    ginv = _mol_geminal_inv(ham, r_up, r_dn)
    RT = np.eye(3)
    ref_R = np.asarray(
        assemble.compute_grad_e_L(ham, jnp.asarray(r_up), jnp.asarray(r_dn), RT, wrt="R", geminal_inv=jnp.asarray(ginv))
    )
    ref_up, ref_dn = assemble.compute_grad_e_L(
        ham, jnp.asarray(r_up), jnp.asarray(r_dn), RT, wrt="r", geminal_inv=jnp.asarray(ginv)
    )
    native = jqmc_rs.vmc_grad_e_L_ecp(wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, r_up, r_dn, _W, _GP, ginv)
    for tag, got, ref in zip(("dR", "dr_up", "dr_dn"), native, (ref_R, np.asarray(ref_up), np.asarray(ref_dn)), strict=True):
        assert np.all(np.isfinite(ref)) and np.max(np.abs(ref)) > 0.0, tag
        np.testing.assert_allclose(np.asarray(got), ref, atol=atol, rtol=rtol, err_msg=f"seed={seed} HF {tag}")


@pytest.mark.light
@pytest.mark.parametrize("seed", [7])
def test_ecp_dtn_hf_stress_open_shell_pbc(pbc_ham, seed):
    """(c) Periodic real-Gamma HF stress, native == XLA, O-lsda 4 up / 2 dn + DTN."""
    ham = pbc_ham
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=seed, scale=0.6, cond_max=15.0)
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn), dtype=np.float64))
    RT = jnp.eye(3)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_FORCE, level="medium")
    ref_eps = np.asarray(
        assemble.compute_grad_e_L(ham, jnp.asarray(r_up), jnp.asarray(r_dn), RT, wrt="eps", geminal_inv=jnp.asarray(g_inv))
    )
    native_eps = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, r_up, r_dn, _W, _GP, g_inv))
    assert ref_eps.shape == (3, 3) and np.all(np.isfinite(ref_eps)) and np.max(np.abs(ref_eps)) > 0.0
    np.testing.assert_allclose(native_eps, ref_eps, atol=atol, rtol=rtol, err_msg=f"seed={seed} HF stress dE_L/d(eps)")

"""Native real ``MCMC.run`` with an INACTIVE (all-zero ``j_matrix``) J3 over a complex plane-wave child.

Issue #157. An all-zero ``j_matrix`` makes the three-body Jastrow factor identically zero, so every
sampled quantity of the native real (Gamma) MCMC route must equal the same run without J3. The
native ``JastrowData`` parser drops such a J3, but the run's observable context re-read the J3
orbital unconditionally in two places:

* ``SweepArgs::from_py_with_cgem`` (the per-run Metropolis sweep of ``run_energy_chunk_batch`` and
  ``mcmc_sweep_batch``);
* ``ParamDerivArgs::from_py`` (the parameter-derivative context, ``build_obs_ctx(...,
  want_param_deriv=True)``).

When the J3 orbital is an MO over a composite of a real pGTO and a COMPLEX plane-wave child, the
real ``OrbData`` parser drops the complex child, so the orbital has fewer AO rows than MO columns
and the sweep / derivative kernels panicked with ``ndarray: index out of bounds``.

Fixture (shared with #140 / PR #143): the committed Li Gamma ccECP cell, a real constant plane-wave
single determinant, and an identity-MO-wrapped (pGTO s, plane-wave) composite J3 orbital with a
zero ``j_matrix``. Children: a complex plane wave with a nonzero G vector (``complex_pw_G``), a
complex constant ``k = 0`` plane wave (``complex_pw_0``), and a real constant plane wave
(``real_pw``) as the control that the real parser represents.

Independent references:

* sampling and value / MO-coefficient derivatives: the same native call with
  ``jastrow_three_body_data=None`` (J3 is identically 0), compared exactly;
* the J3 ``j_matrix`` derivative of an inactive J3 is NOT zero (it equals products of J3 orbital
  values, which the real parser cannot evaluate for the complex children), so the native route
  must refuse it with ``NotImplementedError`` -- the documented hand-off to the JAX route
  (``JQMC_FLAG_RUST_ON_CPU=0``), which serves it -- never a Rust panic and never an omitted block.

Each check is an exact identity or an error disposition judged within the session's precision
mode; no tolerance is involved.
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

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import precision_policy_snapshot
from jqmc.atomic_orbital import AOs_cart_pgto_data, AOs_composite_data, AOs_plane_wave_complex_data, AOs_plane_wave_data
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.jqmc_mcmc import _NV_BASE_MESH, MCMC, Nv_default
from jqmc.atomic_orbital import is_complex_aos
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data
from tests.test_rs_kernel._rs_fixtures import rs_ham

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.with_chdir

_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
_OFFSET_UP = np.array([0.3, 0.2, 0.1])
_OFFSET_DN = np.array([0.5, -0.2, 0.1])
_MCMC_SEED = 34356
_MCMC_STEPS = 4
_CHILDREN = ["complex_pw_G", "complex_pw_0", "real_pw"]
_COMPLEX_CHILDREN = ["complex_pw_G", "complex_pw_0"]
_J3_REFUSAL = "supported only by the complex native path"


def _witness(child: str):
    """(J3 Hamiltonian, J3-free Hamiltonian, r_up, r_dn, geminal inverse) for one J3 child."""
    base = rs_ham("pbc_composite_jnone")
    st = base.structure_data
    gauss = AOs_cart_pgto_data.build(
        structure_data=st,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
        angular_momentums=(0,),
        polynominal_order_x=(0,),
        polynominal_order_y=(0,),
        polynominal_order_z=(0,),
    )
    const_pw = AOs_plane_wave_data.build(st, np.zeros((1, 3)), np.zeros(1))
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(st.cell)).T
    if child == "complex_pw_G":
        j3_child = AOs_plane_wave_complex_data.build(st, np.stack([np.zeros(3), reciprocal[0]]), k_point=(0.0, 0.0, 0.0))
    elif child == "complex_pw_0":
        j3_child = AOs_plane_wave_complex_data.build(st, np.zeros((1, 3)), k_point=(0.0, 0.0, 0.0))
    else:
        j3_child = const_pw
    n_ao = 1 + int(j3_child.num_ao)
    basis = AOs_composite_data.build((gauss, j3_child))
    dtype = np.complex128 if is_complex_aos(basis) else np.float64
    orb = MOs_data.build(aos_data=basis, mo_coefficients=np.eye(n_ao, dtype=dtype))
    j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=np.zeros((n_ao, n_ao + 1)))
    j3.sanity_check()
    det_orb = MOs_data(aos_data=const_pw, num_mo=1, mo_coefficients=np.array([[1.0]]))
    geminal = SD_data(num_electron_up=1, num_electron_dn=1, orb_data_up_spin=det_orb, orb_data_dn_spin=det_orb)

    def ham(three_body):
        jastrow = Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=three_body, jastrow_dtn_data=None
        )
        out = dataclasses.replace(base, wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal))
        out.sanity_check()
        return out

    center = np.asarray(st.positions[0], dtype=np.float64)
    r_up = (center + _OFFSET_UP)[None, :]
    r_dn = (center + _OFFSET_DN)[None, :]
    g = np.asarray(compute_geminal_all_elements(geminal, jnp.asarray(r_up), jnp.asarray(r_dn)), dtype=np.float64)
    return ham(j3), ham(None), r_up, r_dn, np.linalg.inv(g)


def _run_mcmc(ham, *, param_deriv: bool) -> MCMC:
    mcmc = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=_MCMC_SEED,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_stress=False,
        comput_log_WF_param_deriv=param_deriv,
        comput_e_L_param_deriv=param_deriv,
        random_discretized_mesh=False,
    )
    mcmc.run(num_mcmc_steps=_MCMC_STEPS)
    return mcmc


def _pd_ctx(ham):
    wf = ham.wavefunction_data
    return jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        False,
        True,
        False,
        want_force=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_native_mcmc_inactive_complex_child_j3_samples_as_j3_free(child, monkeypatch):
    """Native real energy-only ``MCMC.run`` with an inactive J3 is the J3-free run, exactly.

    Two walkers, four steps, fixed mesh, seed 34356, native backend. The stored ``e_L`` and
    ``w_L`` of every step and the final walker positions equal those of the same run with
    ``jastrow_three_body_data=None``; that run moves the walkers and stores finite nonzero
    energies, so equality is not vacuous. ``real_pw`` is the control.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham_j3, ham_bare, *_ = _witness(child)
    ref = _run_mcmc(ham_bare, param_deriv=False)
    got = _run_mcmc(ham_j3, param_deriv=False)
    ref_e = np.asarray(ref.e_L)
    ref_up = np.asarray(ref._MCMC__latest_r_up_carts)
    assert ref_e.shape == (_MCMC_STEPS, 2)
    assert np.all(np.isfinite(ref_e)) and np.all(ref_e != 0.0)
    np.testing.assert_array_equal(np.asarray(got.e_L), ref_e)
    np.testing.assert_array_equal(np.asarray(got.w_L), np.asarray(ref.w_L))
    np.testing.assert_array_equal(np.asarray(got._MCMC__latest_r_up_carts), ref_up)
    np.testing.assert_array_equal(np.asarray(got._MCMC__latest_r_dn_carts), np.asarray(ref._MCMC__latest_r_dn_carts))
    assert len(np.unique(ref_e)) > 1  # the walkers moved: the stored energies vary over steps / walkers


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_native_param_deriv_ctx_inactive_complex_child_j3_non_j3_blocks_equal_j3_free(child):
    """The native parameter-derivative context treats an inactive J3 as absent for every non-J3 block.

    ``build_obs_ctx(..., want_param_deriv=True)`` with the inactive J3 at one fixed configuration:
    the MO-coefficient O_k (``pd_ok_mo``), its dT/dc (``pd_el_mo``, with and without the geminal
    inverse) and its non-local ECP add (``pd_el_ecp_mo``) equal the J3-free context exactly.
    ``real_pw`` is the control.
    """
    ham_j3, ham_bare, r_up, r_dn, ginv = _witness(child)
    ctx_j3, ctx_bare = _pd_ctx(ham_j3), _pd_ctx(ham_bare)
    calls = {
        "pd_ok_mo": lambda c: c.pd_ok_mo(r_up, r_dn),
        "pd_el_mo": lambda c: c.pd_el_mo(r_up, r_dn),
        "pd_el_mo_ginv": lambda c: c.pd_el_mo(r_up, r_dn, ginv),
        "pd_el_ecp_mo": lambda c: c.pd_el_ecp_mo(r_up, r_dn, _WEIGHTS, _GRID),
    }
    nonzero = 0
    for name, call in calls.items():
        ref = [np.asarray(v) for v in call(ctx_bare)]
        got = [np.asarray(v) for v in call(ctx_j3)]
        assert len(got) == len(ref) == 2, name
        for g, r in zip(got, ref, strict=True):
            assert np.all(np.isfinite(r)), name
            np.testing.assert_array_equal(g, r, err_msg=f"{child}: {name}")
            nonzero += int(np.any(r != 0.0))
    assert nonzero > 0  # at least one compared block is nonzero


@pytest.mark.light
@pytest.mark.parametrize("child", _COMPLEX_CHILDREN)
def test_native_j3_param_deriv_inactive_complex_child_j3_refused_not_panicking(child, monkeypatch):
    """The native J3 ``j_matrix`` derivative of an inactive complex-child J3 is refused loudly.

    The context's J3-family methods (``pd_ok_j3``, ``pd_el_j3``, ``pd_el_ecp_j3`` and their batch
    forms) raise ``NotImplementedError`` naming the complex native path, and so does the public
    native ``MCMC.run`` with parameter derivatives (default mask, J3 ``j_matrix`` included); neither
    panics. The JAX route the message points to (``JQMC_FLAG_RUST_ON_CPU=0``) runs
    the same MCMC and stores a finite, nonzero ``j3_matrix`` O_k block.
    """
    ham_j3, _, r_up, r_dn, _ = _witness(child)
    ctx = _pd_ctx(ham_j3)
    r_up_b, r_dn_b = r_up[None], r_dn[None]
    calls = {
        "pd_ok_j3": lambda: ctx.pd_ok_j3(r_up, r_dn),
        "pd_el_j3": lambda: ctx.pd_el_j3(r_up, r_dn),
        "pd_el_ecp_j3": lambda: ctx.pd_el_ecp_j3(r_up, r_dn, _WEIGHTS, _GRID),
        "pd_ok_j3_batch": lambda: ctx.pd_ok_j3_batch(r_up_b, r_dn_b),
        "pd_el_j3_batch": lambda: ctx.pd_el_j3_batch(r_up_b, r_dn_b),
        "pd_el_ecp_j3_batch": lambda: ctx.pd_el_ecp_j3_batch(r_up_b, r_dn_b, _WEIGHTS, _GRID[None]),
    }
    for name, call in calls.items():
        with pytest.raises(NotImplementedError, match=_J3_REFUSAL):
            call()

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    with pytest.raises(NotImplementedError, match=_J3_REFUSAL):
        _run_mcmc(ham_j3, param_deriv=True)

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    xla = _run_mcmc(ham_j3, param_deriv=True)
    ok_j3 = np.asarray(xla.dln_Psi_dc["j3_matrix"])
    assert ok_j3.shape[:2] == (_MCMC_STEPS, 2)
    assert np.all(np.isfinite(ok_j3)) and np.any(ok_j3 != 0.0)

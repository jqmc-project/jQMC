"""Native vs XLA parity: complex twist, ccECP, active J3, several electrons per spin.

Every other complex-twist ECP parity fixture has one up and one down electron
(``twist_sphepgto_*``, ``pgto_pw_twist_*``, ``bspline_twist_*``, QE Li_k025
projections, ``build_twist_mo_j3_ham``). There, the per-electron offsets of the
native complex J3 ECP paths are all zero: the J3 candidate row of an ECP quadrature
group (``gfmc::compute_ecp_nonlocal_mesh_complex_fast``, ``global_row``), the J3
candidate-row refresh of the hopped electron in the GFMC interval cache (same
function, ``global_off``) and the down-spin J3 drift row of the complex HF force
(``gfmc_forces::gfmc_ecp_nonlocal_force_complex``, ``e - n_up``). An indexing fault
there survived every parity test (issue #651).

The fixture is the committed QE O-lsda_k025 save (triplet oxygen at
k = (1/4, 1/4, 1/4), ccECP, 4 up / 2 dn), converted by the production twist
converter (kind=pgto, J2 pade) with an ACTIVE complex-pGTO J3 attached (the
generated-matrix recipe ``_make_j3``). The checks are

  (a) the fixed-configuration complex local energy, native
      ``compute_local_energy_fast_complex`` (FAST ECP candidate mesh) against the XLA
      ``compute_local_energy_fast`` and the cache-free XLA ``compute_local_energy``;
  (b) the HF force dE_L/d(R, r_up, r_dn), native ``vmc_grad_e_L_pbc_complex`` against
      the XLA ``assemble.compute_grad_e_L``;
  (c) the production ``GFMC_t`` / ``GFMC_n`` dltmove chains, native against XLA on
      the same walk (same seed, random mesh, 2 walkers): Threefry keys, final
      positions, every stored e_L / w_L and the public energy.

The GFMC chain uses ``non_local_move="dltmove"``: the XLA ``GFMC_t`` t-move result
on this wavefunction class is itself wrong (issue #652); the dltmove chains of both
drivers and the ``GFMC_n`` chains are not affected by that defect.
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

import jqmc_rs

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import assemble
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy, compute_local_energy_fast
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests._det_gradlap_chain_fixtures import _o_lsda_twist_ham
from tests.test_jqmc_kernel._generated_matrix_base import _make_j3
from tests.test_rs_kernel.test_gfmc_chain_modes import _assert_deterministic_driver_parity
from tests.test_rs_kernel.test_route_parity_gfmc_complex import _run

pytestmark = [pytest.mark.with_chdir]

_N_UP = 4
_N_DN = 2
_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_ZONES_FORCE = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_MESH = icosahedron_sym_mesh_Nv12
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)
_GFMC_SEED = 34356


@pytest.fixture(scope="module")
def ham_pair(tmp_path_factory):
    """(J2-only, J2 + active complex-pGTO J3) O-lsda_k025 twist hams."""
    ham_j2 = _o_lsda_twist_ham(str(tmp_path_factory.mktemp("o_lsda_k025")))
    gd = ham_j2.wavefunction_data.geminal_data
    assert (int(gd.num_electron_up), int(gd.num_electron_dn)) == (_N_UP, _N_DN)
    assert ham_j2.coulomb_potential_data.ecp_flag, "the O-lsda_k025 fixture must carry the ccECP"
    leaf = gd.orb_data_up_spin
    j3 = _make_j3(getattr(leaf, "aos_data", leaf))
    assert np.max(np.abs(np.asarray(j3.j_matrix))) > 0.0
    jd = ham_j2.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3)
    ham_j3 = ham_j2.replace(wavefunction_data=ham_j2.wavefunction_data.replace(jastrow_data=jd))
    return ham_j2, ham_j3


def _config(ham, seed):
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[0] + rng.normal(scale=0.6, size=(_N_UP, 3))
    r_dn = R[0] + rng.normal(scale=0.6, size=(_N_DN, 3))
    return r_up, r_dn


def _geminal_inv(ham, r_up, r_dn):
    gd = ham.wavefunction_data.geminal_data
    return np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))).astype(np.complex128)


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_fast_complex_twist_multi_electron_ecp_j3(ham_pair, seed):
    """(a) Fixed-configuration complex e_L incl. the native FAST ECP mesh, native == XLA."""
    ham_j2, ham = ham_pair
    r_up, r_dn = _config(ham, seed)
    RT = jnp.eye(3)
    g_inv = _geminal_inv(ham, r_up, r_dn)
    re, im = jqmc_rs.compute_local_energy_fast_complex(ham, r_up, r_dn, _W, _GP, g_inv)
    native = complex(re, im)
    xla_fast = complex(compute_local_energy_fast(ham, r_up, r_dn, RT, jnp.asarray(g_inv), None))
    xla_ref = complex(compute_local_energy(ham, r_up, r_dn, RT))
    xla_j2 = complex(compute_local_energy(ham_j2, r_up, r_dn, RT))
    # Non-vacuity: the J3 changes e_L by far more than the tolerance and the twist e_L is complex.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert abs(xla_ref - xla_j2) > 1.0e3 * (atol + rtol * abs(xla_ref)), (xla_ref, xla_j2)
    assert xla_ref.imag != 0.0, xla_ref
    for tag, ref in (("XLA fast", xla_fast), ("XLA cache-free", xla_ref)):
        for part, a, b in (("Re", native.real, ref.real), ("Im", native.imag, ref.imag)):
            assert np.isclose(a, b, rtol=rtol, atol=atol), (f"seed={seed} {part} e_L native vs {tag}", a, b, abs(a - b))


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1])
def test_hf_force_complex_twist_multi_electron_ecp_j3(ham_pair, seed):
    """(b) HF force dE_L/d(R, r_up, r_dn), native ``vmc_grad_e_L_pbc_complex`` == XLA."""
    _, ham = ham_pair
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    r_up, r_dn = _config(ham, seed)
    g_inv = _geminal_inv(ham, r_up, r_dn)
    RT = jnp.eye(3)
    ref_R = np.asarray(
        assemble.compute_grad_e_L(ham, jnp.asarray(r_up), jnp.asarray(r_dn), RT, wrt="R", geminal_inv=jnp.asarray(g_inv))
    )
    ref_up, ref_dn = assemble.compute_grad_e_L(
        ham, jnp.asarray(r_up), jnp.asarray(r_dn), RT, wrt="r", geminal_inv=jnp.asarray(g_inv)
    )
    native = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, r_up, r_dn, _W, _GP, g_inv)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_FORCE, level="medium")
    atol, rtol = max(atol, ATOL_FLOOR_MEDIUM), max(rtol, RTOL_FLOOR_MEDIUM)
    for tag, got, ref in zip(("dR", "dr_up", "dr_dn"), native, (ref_R, np.asarray(ref_up), np.asarray(ref_dn)), strict=True):
        assert np.all(np.isfinite(ref)) and np.max(np.abs(ref)) > 0.0, tag
        np.testing.assert_allclose(np.asarray(got), ref, atol=atol, rtol=rtol, err_msg=f"seed={seed} HF {tag}")


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_dltmove_chain_complex_twist_multi_electron_ecp_j3(ham_pair, driver):
    """(c) Production GFMC dltmove chain, native == XLA step by step on the same walk."""
    _, ham = ham_pair
    g_xla = _run(driver, ham, _GFMC_SEED, "jax", "dltmove")
    g_rs = _run(driver, ham, _GFMC_SEED, "fast", "dltmove")
    cls = GFMC_t if driver == "t" else GFMC_n
    assert isinstance(g_rs, cls) and isinstance(g_xla, cls)
    assert np.max(np.abs(np.imag(np.asarray(g_xla._stored_e_L)))) > 0.0, "complex e_L leg is vacuous"
    for g in (g_rs, g_xla):
        assert np.all(np.isfinite(np.asarray(g._stored_e_L))) and np.all(np.isfinite(np.asarray(g._stored_w_L)))
    _assert_deterministic_driver_parity(g_rs, g_xla, driver, periodic=True, random_mesh=True)

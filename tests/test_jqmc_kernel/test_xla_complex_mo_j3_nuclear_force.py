"""XLA analytic nuclear force of a complex-MO three-body Jastrow on a twist against AD of e_L (#90).

On a twist (k != 0) the public QE converter's ``-j3 mo`` gives the three-body Jastrow an MO basis
whose coefficients ``C`` are complex (``jastrow_factor._phase_free_j3_orbitals`` keeps the complex
occupied-MO matrix). The J3 orbital is the real field ``Re(C @ AO)``. A complex ``C`` is not folded
to AO rank (``assemble._j3_mo_to_bare_ao``), so the J3 kinetic nuclear force
``force._j3_side`` takes its MO-rank branch. That branch contracted the complex drift cotangents
``bar_gJ = -(gJ + gD)`` (``gD`` is the complex determinant drift) with the complex ``C @ dAO`` and
took ``Re`` only at the end, which leaks ``-Im(bar) . Im(C dAO)``. The analytic force of
``assemble.compute_grad_e_L(wrt="R")`` was off by about 2.5e-3 Ha/bohr while every existing test
passed: the generated twist matrix only reaches a bare-AO or square identity-MO J3 (issue #90).

This module builds a complex-MO J3 on three committed twist fixtures (Li ccECP, k = 0.25):

* ``pgto``: the committed ``lithium_k025.h5`` sphe-pGTO SD with J1/J2, J3 basis = its complex
  ``(1, 26)`` occupied MO (``build_twist_mo_j3_ham``);
* ``pgto_pw`` / ``pgto_bspline``: the committed QE-projected pGTO(+)plane-wave and
  pGTO(+)B-spline composite SDs, J3 basis = their complex occupied MO.

The phase-free MO is the same reference basis the converter uses. The ``j_matrix`` is a seeded
random matrix of scale 0.3 with a symmetric pair block, so the J3 share of the force is a sizeable
part of the total. The reference is reverse-mode AD of ``Re compute_local_energy`` with respect to
the positions of every ``Structure_data`` leaf in the Hamiltonian. It differentiates the scalar
local energy itself and uses no analytic-force code. In full precision it agrees with a
fourth-order central finite difference of the same function (issue #90: 7.8e-11 for the kinetic
part at h = 1e-3). Full keeps that AD comparison. Mixed compares the same analytical nuclear-gradient
route on independently rebuilt full/mixed inputs under the central parameter-derivative
fidelity budget, preserving the original J3-share/nonzero/real/finite controls.
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
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import assemble
from jqmc.hamiltonians import compute_local_energy
from jqmc.jastrow_factor import Jastrow_three_body_data, _phase_free_j3_orbitals
from jqmc.structure import Structure_data
from tests.test_rs_kernel._rs_fixtures import build_twist_mo_j3_ham, draw_well_conditioned_cfg, rs_ham
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._grad_fidelity_harness import _clear_policy_model_caches, assert_parameter_mixed_fidelity

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = [pytest.mark.with_chdir]

#: Zones crossed by the twist local energy with J1/J2/J3 and its nuclear derivative.
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
#: Scale of the seeded random J3 matrix (the converter writes zeros; an optimized J3 is nonzero).
_J3_SCALE = 0.3


def _with_complex_mo_j3(ham, j3_orbital=None):
    """Attach a complex-MO J3 (phase-free occupied MO of the determinant) with a seeded random matrix."""
    wf = ham.wavefunction_data
    if j3_orbital is None:
        j3_orbital = _phase_free_j3_orbitals(wf.geminal_data.orb_data_up_spin)
    coefficients = np.asarray(j3_orbital.mo_coefficients)
    assert np.iscomplexobj(coefficients) and np.max(np.abs(coefficients.imag)) > 1.0e-2, coefficients
    n_mo = coefficients.shape[0]
    rng = np.random.default_rng(7)
    j_matrix = _J3_SCALE * rng.standard_normal((n_mo, n_mo + 1))
    j_matrix[:, :n_mo] = 0.5 * (j_matrix[:, :n_mo] + j_matrix[:, :n_mo].T)
    j3 = Jastrow_three_body_data(orb_data=j3_orbital, j_matrix=j_matrix)
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3)))
    ham.sanity_check()
    return ham


def _hamiltonian(kind):
    if kind == "pgto":
        ham = build_twist_mo_j3_ham()
        return _with_complex_mo_j3(ham, ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data)
    return _with_complex_mo_j3(rs_ham({"pgto_pw": "pgto_pw_twist_jnone", "pgto_bspline": "pgto_bspline_twist_jnone"}[kind]))


def _nuclear_ad(ham, r_up, r_dn):
    """Reverse-mode AD of ``Re e_L`` with every ``Structure_data`` leaf's positions shifted by ``dR``."""
    is_structure = lambda x: isinstance(x, Structure_data)  # noqa: E731

    def energy(dR):
        shifted = jax.tree_util.tree_map(
            lambda x: x.replace(positions=jnp.asarray(x.positions) + dR) if is_structure(x) else x,
            ham,
            is_leaf=is_structure,
        )
        return jnp.real(compute_local_energy(shifted, r_up, r_dn, jnp.eye(3)))

    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions, dtype=jnp.float64))
    return np.asarray(jax.grad(energy)(zero), dtype=np.float64)


def _force_fidelity_case(kind):
    _clear_policy_model_caches()
    ham = _hamiltonian(kind)
    j3_orbital = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    assert type(j3_orbital).__name__ == "MOs_data" and np.iscomplexobj(np.asarray(j3_orbital.mo_coefficients))
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=1)
    return ham, jnp.asarray(r_up), jnp.asarray(r_dn)


def _analytical_force_outputs(case):
    ham, r_up, r_dn = case
    wf = ham.wavefunction_data
    no_j3 = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=None)))
    return {
        "actual": np.asarray(assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="R")),
        "no_j3": np.asarray(assemble.compute_grad_e_L(no_j3, r_up, r_dn, jnp.eye(3), wrt="R")),
        "r_up": np.asarray(r_up),
        "r_dn": np.asarray(r_dn),
    }


def _check_force_fidelity(kind):
    mixed, full = mixed_vs_full(lambda: _force_fidelity_case(kind), _analytical_force_outputs)
    assert np.array_equal(mixed["r_up"], full["r_up"]) and np.array_equal(mixed["r_dn"], full["r_dn"]), (
        "full/mixed nuclear fixture coordinates differ"
    )
    actual, reference = mixed["actual"], full["actual"]
    assert not np.iscomplexobj(actual) and np.all(np.isfinite(actual)), actual
    assert np.all(np.isfinite(reference)), reference
    j3_share = reference - full["no_j3"]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.any(np.abs(j3_share) > atol + rtol * np.abs(reference)), j3_share
    assert_parameter_mixed_fidelity(actual, reference, f"{kind} complex-MO J3 nuclear gradient mixed vs full")


# Every row takes 45-100 s on the CPU reference host (local-energy AD), above the 60 s light budget.
@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("pgto", marks=pytest.mark.heavy),
        pytest.param("pgto_pw", marks=pytest.mark.heavy),
        pytest.param("pgto_bspline", marks=pytest.mark.heavy),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_xla_complex_mo_j3_nuclear_force_matches_local_energy_ad(kind):
    """Full retains nuclear AD; mixed analytical nuclear derivatives retain independently rebuilt full fidelity."""
    if is_mixed_precision_enabled():
        _check_force_fidelity(kind)
        return
    ham = _hamiltonian(kind)
    j3_orbital = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    assert type(j3_orbital).__name__ == "MOs_data" and np.iscomplexobj(np.asarray(j3_orbital.mo_coefficients))
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=1)
    r_up, r_dn = jnp.asarray(r_up), jnp.asarray(r_dn)

    reference = _nuclear_ad(ham, r_up, r_dn)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.all(np.isfinite(reference)), reference
    # Non-vacuous: dropping the J3 share of the reference force would fail the comparison below.
    wf = ham.wavefunction_data
    no_j3 = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=None)))
    j3_share = reference - _nuclear_ad(no_j3, r_up, r_dn)
    assert np.any(np.abs(j3_share) > atol + rtol * np.abs(reference)), j3_share

    actual = np.asarray(assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="R"))
    assert not np.iscomplexobj(actual) and np.all(np.isfinite(actual)), actual
    np.testing.assert_allclose(
        actual,
        reference,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"XLA analytic d e_L/dR != AD of Re e_L with a complex-MO J3 on a twist ({kind})",
    )

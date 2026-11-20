"""Analytical ECP derivatives with a non-local radial power other than 2 (issue 572).

A non-local ECP term (l, c, a, n) has the radial factor ``V(r) = c r^(n-2) exp(-a r^2)``, whose
derivative is ``c [(n-2) r^(n-3) - 2 a r^(n-1)] exp(-a r^2)``. Every committed ECP fixture stores
``n = 2`` for the non-local channels, where the first term vanishes, so the analytical electron
gradient, nuclear force, molecular strain and ``dltmove`` correction derivatives
(``jqmc.derivatives.ecp_force``) were never evaluated with that term alive.

The input is the committed N2 ccECP cc-pVTZ wavefunction with a J1 + J2 Jastrow, whose non-local
term powers are replaced by ``n = 0`` or ``n = 3`` (the local channel keeps its stored powers).
References, both evaluated in the active precision mode:

* the local-energy derivatives are compared with JAX autodiff of the production local energy
  (``jqmc.derivatives._auto``), whose forward non-local ECP routes honour the stored power
  (MATH-ECP-NONLOCAL-RADIAL-POWER-ROUTES-001);
* the ``dltmove`` correction ``delta = sum_g max(V^det_g, 0) (1 - jr_g)`` is rebuilt here from
  the public nearest-neighbour non-local evaluator (determinant-only ratio) and the Jastrow
  value, and differentiated by JAX autodiff.
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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

from jqmc import derivatives
from jqmc._precision import get_tolerance_cancelling_mean, get_tolerance_min, is_mixed_precision_enabled
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.derivatives import _auto, ecp_force
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data, compute_Jastrow_part
from jqmc.jqmc_mcmc import _extract_active_positions_from_ham, _inject_active_positions
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_STEM = "N2_ecp_ccpvtz_cart.h5"
_POWERS = (0, 3)
_AXES = ("r", "R", "eps")
# Same zone set as the committed analytical-vs-autodiff ECP local-energy tests.
_E_L_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "local_energy")
_DLTMOVE_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb")
# A tilted quadrature rotation (Rodrigues, axis (1, 2, 3), angle 0.4 rad) keeps the mesh off the
# symmetry axes of the molecule.
_AXIS = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
_K = np.array([[0.0, -_AXIS[2], _AXIS[1]], [_AXIS[2], 0.0, -_AXIS[0]], [-_AXIS[1], _AXIS[0], 0.0]])
_RT = np.eye(3) + np.sin(0.4) * _K + (1.0 - np.cos(0.4)) * (_K @ _K)


def _assert_derivative_block(actual: np.ndarray, reference: np.ndarray, zones: tuple[str, ...]) -> None:
    """Strict zone tolerance for one homogeneous derivative block (same-precision routes).

    Full compares element by element. In mixed both routes cross fp32 forward zones, so their
    disagreement is set by the block's largest term (about one fp32 eps of ``max|reference|``),
    not by an element that happens to cancel; the strict pair is therefore rebased once onto
    ``max|reference|`` (``get_tolerance_cancelling_mean``). The dropped radial-power term moves
    every block by several percent of that scale or more, far outside either envelope.
    """
    assert actual.shape == reference.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    atol, rtol = get_tolerance_min(zones, "strict")
    if is_mixed_precision_enabled():
        atol, rtol = get_tolerance_cancelling_mean(atol, rtol, np.max(np.abs(reference)))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)


def _build_ham(power: int) -> Hamiltonian_data:
    """N2 ccECP (J1 + J2) with every non-local term's radial power replaced by ``power``."""
    structure, _aos, _mos_up, _mos_dn, geminal, coulomb = read_trexio_file(os.path.join(_TREXIO, _STEM), store_tuple=True)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure,
        core_electrons=core_electrons_of(coulomb, len(structure.positions)),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure)
    jastrow = Jastrow_data(
        jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None, jastrow_dtn_data=None
    )
    powers = list(coulomb.powers)
    for index in coulomb._non_local_part_index:
        powers[int(index)] = power
    coulomb = coulomb.replace(powers=tuple(powers))
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal),
    )
    # Witness: the padded non-local table read by ecp_force carries the replaced power on every
    # live term, and the local channel is untouched.
    _ang, _exp, coef, pow_padded = coulomb._padded_parameters_tuple
    live = np.asarray(coef) != 0.0
    assert live.any()
    assert np.all(np.asarray(pow_padded)[live] == power)
    assert set(int(p) for p in coulomb._powers_local_part) == {1, 2, 3}
    return ham


def _draw(ham: Hamiltonian_data):
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(0)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(int(gd.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.7, size=(int(gd.num_electron_dn), 3)))
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("axis", _AXES)
@pytest.mark.parametrize("power", _POWERS)
def test_ecp_local_energy_derivatives_match_autodiff_at_nonlocal_power(power, axis):
    """Analytical d e_L / d{r, R, eps} (ECP local + non-local, molecular) equals JAX autodiff of
    the production local energy when the non-local radial power is ``power != 2``."""
    ham = _build_ham(power)
    r_up, r_dn = _draw(ham)
    RT = jnp.asarray(_RT)
    actual = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, {f"opt_{axis}": True}, analytic=True)[axis]
    reference = _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt=axis)
    actual_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(actual)]
    reference_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(reference)]
    assert len(actual_leaves) == len(reference_leaves)
    for got, want in zip(actual_leaves, reference_leaves):
        _assert_derivative_block(got, want, _E_L_ZONES)


def _dltmove_delta_reference(ham: Hamiltonian_data, r_up: jax.Array, r_dn: jax.Array, RT: jax.Array) -> jax.Array:
    """``delta = sum_g max(V^det_g, 0) (1 - jr_g)`` from the public nearest-neighbour evaluator."""
    wf = ham.wavefunction_data
    r_up_mesh, r_dn_mesh, v_det, _ = compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data, wf, r_up, r_dn, RT=RT, flag_determinant_only=True
    )
    j_ref = compute_Jastrow_part(wf.jastrow_data, r_up, r_dn)
    j_mesh = jax.vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(wf.jastrow_data, r_up_mesh, r_dn_mesh)
    jr = jnp.real(jnp.exp(j_mesh - j_ref))
    return jnp.sum(jnp.maximum(jnp.real(v_det), 0.0) * (1.0 - jr))


@pytest.mark.light
@pytest.mark.parametrize("axis", _AXES)
@pytest.mark.parametrize("power", _POWERS)
def test_ecp_dltmove_delta_derivatives_match_autodiff_at_nonlocal_power(power, axis):
    """Closed-form d delta / d{r, R, eps} of the molecular ``dltmove`` correction
    (``ecp_force.grad_V_ecp_nl_dltmove``) equals JAX autodiff of the independently rebuilt
    ``delta`` when the non-local radial power is ``power != 2``."""
    ham = _build_ham(power)
    r_up, r_dn = _draw(ham)
    RT = jnp.asarray(_RT)
    # Non-vacuity: some localised (V^det >= 0) candidates carry a Jastrow ratio != 1.
    delta = float(_dltmove_delta_reference(ham, r_up, r_dn, RT))
    assert abs(delta) > get_tolerance_min(_DLTMOVE_ZONES, "strict")[0]

    actual = np.asarray(ecp_force.grad_V_ecp_nl_dltmove(ham, r_up, r_dn, RT, wrt=axis))
    if axis == "r":
        g_up, g_dn = jax.jit(jax.grad(lambda a, b: _dltmove_delta_reference(ham, a, b, RT), argnums=(0, 1)))(r_up, r_dn)
        reference = np.concatenate([np.asarray(g_up), np.asarray(g_dn)], axis=0)
    elif axis == "R":
        active = _extract_active_positions_from_ham(ham)
        grad = jax.jit(jax.grad(lambda act: _dltmove_delta_reference(_inject_active_positions(ham, act), r_up, r_dn, RT)))(
            active
        )
        reference = np.asarray(grad["positions"])
    else:

        def strained(eps: jax.Array) -> jax.Array:
            F = jnp.eye(3, dtype=jnp.float64) + eps
            return _dltmove_delta_reference(ham._apply_strain(F), r_up @ F.T, r_dn @ F.T, RT)

        reference = np.asarray(jax.jit(jax.grad(strained))(jnp.zeros((3, 3), dtype=jnp.float64)))
    _assert_derivative_block(actual, reference, _DLTMOVE_ZONES)

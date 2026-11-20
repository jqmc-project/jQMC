"""Independent Gaussian/J2/ECP formulas for tmove and determinant-localized tmove."""

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

import inspect
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc as gfmc
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from tests.test_jqmc_kernel.test_mcmc_gaussian_energy_reference import (
    _gaussian_hamiltonian,
    _gaussian_orbitals,
    _kinetic_elements,
    _s_orbital_log_derivatives,
    _zone_cast,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _ecp_j2_hamiltonian(sign, b):
    ham = _gaussian_hamiltonian(2, 0.5)
    structure = ham.structure_data
    coulomb = Coulomb_potential_data(
        structure_data=structure,
        ecp_flag=True,
        z_cores=(0.0,),
        max_ang_mom_plus_1=(1,),
        num_ecps=2,
        nucleus_index=(0, 0),
        ang_moms=(0, 1),
        coefficients=(sign * 1.3, 2.0),
        exponents=(0.7, 0.4),
        powers=(2, 1),
    )
    coulomb.sanity_check()
    j2 = Jastrow_two_body_data(jastrow_2b_param=b, jastrow_2b_type="pade", structure_data=structure)
    return ham.replace(
        coulomb_potential_data=coulomb,
        wavefunction_data=ham.wavefunction_data.replace(jastrow_data=Jastrow_data(jastrow_two_body_data=j2)),
    )


def _ecp_reference(points, sign, b, move):
    """Differentiate the Gaussian/J2 formulas and sum the independently authored discrete ECP rule.

    Precision: every piece is evaluated in the dtype of the production zone that
    computes it, with geometry reconstructed in fp64 first: orbital values in
    ao_eval and orbital derivatives in ao_grad_lap, J2 values in jastrow_eval
    and J2 gradient/Laplacian in jastrow_grad_lap, determinant ratios in
    det_ratio, exp(J'-J) in jastrow_ratio, Coulomb/ECP radial terms in coulomb.
    """
    alpha, alat = 0.5, 0.3
    radii = np.linalg.norm(points, axis=-1)
    grad_d, lap_d = _s_orbital_log_derivatives(points, alpha)
    j_dtype = get_dtype_np("jastrow_grad_lap")
    diff = _zone_cast(points[0] - points[1], "jastrow_grad_lap")
    distance = np.linalg.norm(diff)
    first = j_dtype(0.5) / (j_dtype(1) + j_dtype(b) * distance) ** 2
    second = -j_dtype(b) / (j_dtype(1) + j_dtype(b) * distance) ** 3
    grad_j = np.stack([first * diff / distance, -first * diff / distance])
    lap_j = np.full(2, second + j_dtype(2) * first / distance)
    grad_psi = _zone_cast(grad_j, "wf_kinetic") + _zone_cast(grad_d, "wf_kinetic")
    lap_psi = _zone_cast(lap_j, "wf_kinetic") + _zone_cast(lap_d, "wf_kinetic")
    kinetic = np.sum(_kinetic_elements(grad_psi, lap_psi))
    coulomb_points = _zone_cast(points, "coulomb")
    coulomb_radii = np.linalg.norm(coulomb_points, axis=-1)
    local = (
        kinetic
        + 1 / np.linalg.norm(coulomb_points[0] - coulomb_points[1])
        + np.sum(2 * (np.exp(-0.4 * coulomb_radii**2) - 1) / coulomb_radii)
    )
    dtype = get_dtype_np("jastrow_eval")

    def j_value(new_points):
        d = np.linalg.norm(_zone_cast(new_points[..., 0, :] - new_points[..., 1, :], "jastrow_eval"), axis=-1)
        return d / dtype(2) * (dtype(1) + dtype(b) * d) ** dtype(-1)

    baseline = j_value(points)
    directions = np.concatenate([np.eye(3), -np.eye(3)])
    # Cartesian icosahedron: two poles and two staggered rings, each weight 1/12.
    # No coordinates/weights are imported from the production mesh.
    angles = np.arange(5) * 2 * np.pi / 5
    ring_radius = 2 / np.sqrt(5)
    ecp_directions = np.concatenate(
        [
            [[0, 0, 1], [0, 0, -1]],
            np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles), np.full(5, 1 / np.sqrt(5))]),
            np.column_stack(
                [
                    ring_radius * np.cos(angles + np.pi / 5),
                    ring_radius * np.sin(angles + np.pi / 5),
                    np.full(5, -1 / np.sqrt(5)),
                ]
            ),
        ]
    )
    kinetic_off, diagonal_ecp, off_ecp = 0.0, 0.0, 0.0
    spreads = []
    orbital_values = _zone_cast(_gaussian_orbitals(points, alpha, "ao_eval")["s"], "det_ratio")

    def determinant_ratio(moved, electron):
        moved_value = _gaussian_orbitals(moved[:, electron, :], alpha, "ao_eval")["s"]
        return _zone_cast(moved_value, "det_ratio") / orbital_values[electron]

    def jastrow_ratio(moved):
        return np.exp(_zone_cast(j_value(moved) - baseline, "jastrow_ratio"))

    for electron in range(2):
        moved = np.broadcast_to(points, (6, 2, 3)).copy()
        moved[:, electron, :] += alat * directions
        kinetic_off -= np.sum(determinant_ratio(moved, electron) * jastrow_ratio(moved)) / (2 * alat**2)
        moved = np.broadcast_to(points, (12, 2, 3)).copy()
        moved[:, electron, :] = radii[electron] * ecp_directions
        # On a sphere centred on this s orbital the determinant ratio is
        # analytically one; it is still evaluated from ao_eval values.
        mesh_ratio = determinant_ratio(moved, electron) * jastrow_ratio(moved)
        determinant_only = determinant_ratio(moved, electron)
        radial = sign * 1.3 * np.exp(-0.7 * coulomb_radii[electron] ** 2)
        spreads.append(float(np.ptp(jastrow_ratio(moved))))
        diagonal_ecp += max(radial, 0) * np.mean(mesh_ratio if move == "tmove" else determinant_only)
        off_ecp += min(radial, 0) * np.mean(mesh_ratio)
    return np.array([local - kinetic_off + diagonal_ecp, kinetic_off + off_ecp, local + diagonal_ecp + off_ecp]), spreads


def _compare_ecp(sign, b, move, *, closure=None):
    ham = _ecp_j2_hamiltonian(sign, b)
    r_up_carts = jnp.asarray([[0.4, 0.7, -0.2]])
    r_dn_carts = jnp.asarray([[-0.6, -0.3, 0.8]])
    references, spreads = _ecp_reference(np.concatenate([r_up_carts, r_dn_carts]), sign, b, move)
    assert all(value > 0 for value in spreads), "both spins must have an active angle-dependent Jastrow ratio"
    closure = gfmc._compute_V_elements_t if closure is None else closure
    values = closure(ham, r_up_carts, r_dn_carts, jnp.eye(3), move, 0.3)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio"), "strict")
    for quantity, actual, reference in zip(
        ("propagation-diagonal", "propagation-off-diagonal", "measurement-energy"), values, references, strict=True
    ):
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("b", [0.4, 1.2])
@pytest.mark.parametrize("move", ["tmove", "dltmove"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_ecp_jastrow_propagation_matches_closed_form(sign, b, move):
    """The tmove/dltmove diagonal, off-diagonal and measurement use the prescribed Jastrow factors.

    Synthetic parameter-set, no TREXIO: molecular He with one up/one down electron,
    closed-shell restricted SD/MO, centred Cartesian s Gaussian exponent 0.5,
    J2 Pade b=0.4/1.2, no J1/J3/DTN. A zero-core ECP has local
    2*exp(-0.4*r^2)/r and s-channel +/-1.3*exp(-0.7*r^2). Positions are
    (0.4,0.7,-0.2) and (-0.6,-0.3,0.8), a single explicit configuration with
    no conditioning screen or random selection. The geminal is 1 by 1.

    The reference differentiates the Gaussian times exp(r12/(2*(1+b*r12)))
    analytically and independently sums the kinetic six-neighbour rule (alat=0.3)
    and the default twelve-point icosahedron ECP rule, both unrotated. Each
    piece is evaluated in its production zone dtype (orbital values in
    ao_eval, orbital derivatives in ao_grad_lap, J2 in jastrow_eval and its
    gradient/Laplacian in jastrow_grad_lap, ratios in det_ratio/jastrow_ratio,
    Coulomb/ECP in coulomb), never as an fp64 reference for mixed. It imports
    no production energy, ratio, quadrature or debug implementation. Both spins
    have nonconstant Jastrow ratios. Negative ECP channels carry that ratio in
    both methods; positive channels carry it only for tmove, not dltmove.
    Check all three outputs with unchanged central same-mode strict tolerances.
    This is a discrete-operator composition guarantee, not a ground-state estimate,
    exact continuous angular integration, or statistical acceptance test.
    """
    _compare_ecp(sign, b, move)


@pytest.mark.parametrize("fault", ["dlt-positive-localization", "negative-jastrow-factor", "negative-channel-clipping"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_ecp_reference_rejects_actual_localization_fault(fault):
    """After a passing baseline, wrong positive localization, lost Jastrow or reversed clipping fails."""
    sign, move, old, new = {
        "dlt-positive-localization": (1, "dltmove", "flag_determinant_only=True,", "flag_determinant_only=False,"),
        "negative-jastrow-factor": (
            -1,
            "dltmove",
            "V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio",
            "V_nonlocal_FN = V_nonlocal_FN",
        ),
        "negative-channel-clipping": (
            -1,
            "tmove",
            "V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)",
            "V_nonlocal_FN = jnp.maximum(V_nonlocal, 0.0)",
        ),
    }[fault]
    _compare_ecp(sign, 0.4, move)
    source = textwrap.dedent(inspect.getsource(gfmc._compute_V_elements_t))
    assert source.count(old) == (2 if fault == "negative-channel-clipping" else 1)
    namespace = dict(vars(gfmc))
    exec(compile(source.replace(old, new), "<ecp-localization-fault>", "exec"), namespace)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        _compare_ecp(sign, 0.4, move, closure=namespace["_compute_V_elements_t"])

"""Independent nodal-Gaussian anchors for GFMC fixed-node and novar propagation."""

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

import ast
import inspect
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from tests.test_jqmc_kernel.test_mcmc_gaussian_energy_reference import (
    _gaussian_hamiltonian,
    _gaussian_orbitals,
    _kinetic_elements,
    _zone_cast,
)
from jqmc.atomic_orbital import AOs_cart_gto_data
from jqmc.molecular_orbital import MOs_data
from jqmc.determinant import SD_data
from jqmc import jqmc_gfmc as gfmc

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _nodal_hamiltonian(exponent):
    """He triplet SD from s and px orbitals; no Jastrow or ECP and no imported physical values."""
    ham = _gaussian_hamiltonian(2, exponent)
    aos = AOs_cart_gto_data.build(
        structure_data=ham.structure_data,
        nucleus_index=(0, 0, 0, 0),
        num_ao=4,
        num_ao_prim=4,
        orbital_indices=(0, 1, 2, 3),
        exponents=(exponent,) * 4,
        coefficients=(1.0,) * 4,
        angular_momentums=(0, 1, 1, 1),
        polynominal_order_x=(0, 1, 0, 0),
        polynominal_order_y=(0, 0, 1, 0),
        polynominal_order_z=(0, 0, 0, 1),
    )
    up = MOs_data(num_mo=2, aos_data=aos, mo_coefficients=np.eye(4)[:2])
    down = MOs_data(num_mo=0, aos_data=aos, mo_coefficients=np.empty((0, 4)))
    geminal = SD_data(num_electron_up=2, num_electron_dn=0, orb_data_up_spin=up, orb_data_dn_spin=down)
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))


def _slater_rows(orbitals, zone):
    """Return the s and p_x columns of a two-electron Slater matrix cast to ``zone``."""
    return _zone_cast(orbitals["s"], zone), _zone_cast(orbitals["px"], zone)


def _closed_form(points, exponent, alat):
    """Evaluate polynomial-Gaussian ratios, kinetic/Coulomb terms and the prescribed FN regularization.

    The determinant is s(r_1) p_x(r_2) - p_x(r_1) s(r_2), mathematically a nonzero
    constant times (x_2-x_1) exp(-alpha sum_i |r_i|^2). It is expanded
    explicitly from orbital pieces evaluated in their production zone dtypes:
    orbital values in ao_eval, orbital gradients/Laplacians in ao_grad_lap,
    determinant ratios in det_ratio, ln-determinant derivatives in det_grad_lap,
    kinetic assembly in wf_kinetic and Coulomb terms in coulomb. In full mode all
    of these are fp64 and the expansion equals the closed form to fp64 rounding.
    """
    shifts = alat * np.concatenate([np.eye(3), -np.eye(3)])
    s, px = _slater_rows(_gaussian_orbitals(points, exponent, "ao_eval"), "det_ratio")
    determinant = s[0] * px[1] - px[0] * s[1]
    moved_s, moved_px = _slater_rows(_gaussian_orbitals(points[:, None, :] + shifts[None], exponent, "ao_eval"), "det_ratio")
    ratios = np.stack(
        [
            (moved_s[0] * px[1] - moved_px[0] * s[1]) / determinant,
            (s[0] * moved_px[1] - px[0] * moved_s[1]) / determinant,
        ]
    )
    transitions = -ratios / (2 * alat**2)

    # Row-replacement (Laplace) expansion of the derivative of each electron's row.
    s, px = _slater_rows(_gaussian_orbitals(points, exponent, "ao_eval"), "det_grad_lap")
    determinant = s[0] * px[1] - px[0] * s[1]
    derivatives = _gaussian_orbitals(points, exponent, "ao_grad_lap")
    s_grad, px_grad = _zone_cast(derivatives["s_grad"], "det_grad_lap"), _zone_cast(derivatives["px_grad"], "det_grad_lap")
    s_lap, px_lap = _zone_cast(derivatives["s_lap"], "det_grad_lap"), _zone_cast(derivatives["px_lap"], "det_grad_lap")
    grad = np.stack([s_grad[0] * px[1] - px_grad[0] * s[1], s[0] * px_grad[1] - px[0] * s_grad[1]]) / determinant
    lap = np.stack([s_lap[0] * px[1] - px_lap[0] * s[1], s[0] * px_lap[1] - px[0] * s_lap[1]]) / determinant
    kinetic = _kinetic_elements(grad, lap - np.sum(grad**2, axis=1))

    coulomb_points = _zone_cast(points, "coulomb")
    radial = np.linalg.norm(coulomb_points, axis=1)
    ei = -2 / radial
    ei_floor = -2 / np.maximum(radial, alat)
    ee = 1 / np.linalg.norm(coulomb_points[0] - coulomb_points[1])
    discrete_kinetic = np.sum(transitions + 1 / (2 * alat**2), axis=1)
    correction = ei + kinetic - discrete_kinetic
    flips = np.any(ratios <= 0, axis=1)
    clamped = flips & (ei_floor > correction)
    diagonal = 6 / alat**2 + ee + np.sum(np.where(clamped, ei_floor, correction)) + np.maximum(transitions, 0).sum()
    off_diagonal = np.minimum(transitions, 0).sum()
    measurement = ee + np.sum(ei + kinetic)
    return np.array([diagonal, off_diagonal, measurement]), flips, clamped


def _compare_nodal(exponent, alat, *, closure=None):
    ham = _nodal_hamiltonian(exponent)
    r_up_carts = jnp.asarray([[0.4, 0.7, -0.2], [-0.6, -0.3, 0.8]])
    r_dn_carts = jnp.empty((0, 3))
    references, flips, clamped = _closed_form(np.asarray(r_up_carts), exponent, alat)
    closure = gfmc._compute_V_elements_t if closure is None else closure
    values = closure(ham, r_up_carts, r_dn_carts, jnp.eye(3), "tmove", alat)
    atol, rtol = get_tolerance_min(("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio"), "strict")
    for quantity, actual, reference in zip(
        ("propagation-diagonal", "fixed-node-off-diagonal", "continuum-measurement"), values, references, strict=True
    ):
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
    return int(np.count_nonzero(flips)), int(np.count_nonzero(clamped))


@pytest.mark.parametrize(
    "exponent,alat,expected_flips,expected_clamps",
    [(0.5, 0.3, 0, 0), (0.5, 1.5, 2, 1), (0.5, 3.0, 2, 2), (1.25, 0.3, 0, 0), (1.25, 1.5, 2, 0), (1.25, 3.0, 2, 0)],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_nodal_gaussian_fixed_node_and_novar_formula(exponent, alat, expected_flips, expected_clamps):
    """The real GFMC closure obeys an independent nodal Gaussian/Coulomb reference.

    Synthetic open He has two up and zero down electrons (open shell), SD/MO,
    Cartesian s+px occupied orbitals with common exponent 0.5/1.25, no Jastrow
    and no ECP. The s/px/py/pz AO bank and identity MO rows are built in Python,
    not read from TREXIO. Positions are exactly (0.4,0.7,-0.2) and
    (-0.6,-0.3,0.8) Bohr, one configuration, with no cond(G) limit or screening.
    The normalized determinant is proportional to (x2-x1)*exp(-a*sum(r_i^2)).
    Its explicit two-row expansion gives the ratios and per-electron Laplacians;
    with elementary Coulomb interactions they determine all reference values
    without calling production AO, determinant, kinetic, potential or
    propagation helpers. Each piece is evaluated in its production
    zone dtype (orbital values in ao_eval, orbital derivatives in ao_grad_lap,
    ratios in det_ratio, Coulomb in coulomb), never as an fp64 reference for mixed.

    The identity six-neighbour mesh at alat=0.3/1.5/3.0 exercises no crossing,
    node crossing, and zero/one/two active Coulomb cutoffs. Compare propagation
    diagonal, FN off-diagonal and unregularized continuum measurement separately
    with the existing same-mode strict zones. The fixed-node correction must
    not leak into novar=0 measurement. These finite-input equations do not
    assert a ground-state energy, a sampled distribution, or agreement in sigma.
    """
    flips, clamped = _compare_nodal(exponent, alat)
    assert flips == expected_flips
    assert clamped == expected_clamps


@pytest.mark.parametrize("fault", ["sign-flip", "cutoff", "measurement-routing"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_nodal_reference_rejects_actual_propagation_fault(fault):
    """After a passing baseline, omitted sign-flip, reversed cutoff or wrong novar output fails the formula."""
    _compare_nodal(0.5, 1.5)
    source = textwrap.dedent(inspect.getsource(gfmc._compute_V_elements_t))
    if fault == "sign-flip":
        old = "diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))"
        assert source.count(old) == 1
        source = source.replace(old, "diagonal_kinetic_part_SP = 0.0")
    elif fault == "cutoff":
        for spin in ("up", "dn"):
            old = f"diagonal_bare_coulomb_part_el_ion_max_{spin} = jnp.maximum("
            assert source.count(old) == 1
            source = source.replace(old, old.replace("jnp.maximum", "jnp.minimum"))
    else:
        old = "return (V_diag, V_nondiag, e_L_meas)"
        assert source.count(old) == 1
        source = source.replace(old, "return (V_diag, V_nondiag, V_diag + V_nondiag)")
    namespace = dict(vars(gfmc))
    exec(compile(ast.parse(source), "<nodal-gfmc-fault>", "exec"), namespace)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        _compare_nodal(0.5, 1.5, closure=namespace["_compute_V_elements_t"])

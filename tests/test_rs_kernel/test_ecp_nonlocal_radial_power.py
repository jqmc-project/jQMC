"""Native non-local ECP radial power: c r^(n-2) exp(-a r^2) for n != 2.

Every committed ECP fixture (ccECP) stores the non-local power n = 2, so the
native radial factor ``c * r^(n-2) * exp(-a r^2)`` is ``c * exp(-a r^2)`` on all
of them and a kernel that ignores the stored power is indistinguishable from the
correct one (issue #568). These rows feed n in {0, 1, 3} (and the n = 2 control)
through the native VMC energy and LRDMC mesh kernels on a one-nucleus input
whose non-local energy has a closed form, and compare each kernel with both the
closed form and the same-mode XLA route.

Input: one pseudized H centre at the origin, a nodeless s Gaussian SD with one
up and one down electron, no Jastrow, a single non-local l=0 term
``(c, a, n) = (+-1.3, 0.7, n)`` and the local tag l=1 ``(1.0, 0.4, 1)``. The
quadrature points of each electron lie on its own sphere about the only nucleus,
so the orbital ratio is 1, P_0 = 1 and the weights sum to 1: the non-local energy
is ``sum_i c r_i^(n-2) exp(-a r_i^2)``. A second Hamiltonian with ``c = 0``
isolates the non-local contribution of the energy kernels; every other term is
the same arithmetic in both evaluations. Both electrons are within 0.8 bohr of
the nucleus, where the non-local term is O(1) Ha (the default parity scatter
leaves it at ~3e-4 Ha).
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

import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency
from jqmc.atomic_orbital import AOs_sphe_gto_data
from jqmc.coulomb_potential import (
    Coulomb_potential_data,
    compute_ecp_non_local_parts_nearest_neighbors,
    icosahedron_sym_mesh_Nv12,
)
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy, compute_local_energy_fast
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import _compute_V_elements_t
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_ELECTRONS = np.asarray([[0.31, -0.27, 0.49], [-0.37, 0.26, 0.58]])
_EXPONENT = 0.7
_COEFFICIENT = 1.3
_ALAT = 0.30
_MESH_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_MESH_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)

# The closed form is reached through orbital values (ao_eval), the determinant
# ratio (det_ratio) and the projector (coulomb).
_ZONES_CLOSED_FORM = ("coulomb", "det_ratio", "ao_eval")
_ZONES_ECP_NONLOCAL = ("coulomb", "det_ratio", "ao_eval")
_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "det_ratio")
_ZONES_GFMC = ("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio")


def _hamiltonian(power: int, coefficient: float) -> Hamiltonian_data:
    structure = Structure_data.build(
        positions=np.zeros((1, 3)), atomic_numbers=(1,), element_symbols=("H",), atomic_labels=("H1",), pbc_flag=False
    )
    orbital = AOs_sphe_gto_data.build(
        structure_data=structure,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.asarray([0.8]),
        coefficients=np.ones(1),
        angular_momentums=(0,),
        magnetic_quantum_numbers=(0,),
    )
    mos = MOs_data.build(aos_data=orbital, mo_coefficients=np.ones((1, 1)))
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=1, mos_up=mos, mos_dn=mos, time_reversal=False)
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
    coulomb = Coulomb_potential_data(
        structure_data=structure,
        ecp_flag=True,
        z_cores=(0.0,),
        max_ang_mom_plus_1=(1,),
        num_ecps=2,
        nucleus_index=(0, 0),
        ang_moms=(0, 1),
        coefficients=(coefficient, 1.0),
        exponents=(_EXPONENT, 0.4),
        powers=(power, 1),
    )
    coulomb.sanity_check()
    assert list(coulomb._powers_non_local_part) == [power]
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)


def _configuration(ham):
    r_up = np.ascontiguousarray(_ELECTRONS[:1])
    r_dn = np.ascontiguousarray(_ELECTRONS[1:])
    g = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_dn))
    assert g.shape == (1, 1) and np.all(np.isfinite(g)) and g[0, 0] != 0
    assert np.linalg.cond(g) == 1
    return r_up, r_dn, np.linalg.inv(g)


def _radial_factors(power: int, coefficient: float) -> np.ndarray:
    """Per-electron ``c r^(n-2) exp(-a r^2)``, the l=0 projector on a ratio-1 orbital."""
    radius = np.linalg.norm(_ELECTRONS, axis=1)
    return coefficient * radius ** (power - 2) * np.exp(-_EXPONENT * radius**2)


def _assert_power_is_live(power: int, coefficient: float, atol: float, rtol: float) -> None:
    """At n != 2 the closed form must differ from the n = 2 value beyond the tolerance."""
    if power == 2:
        return
    value = _radial_factors(power, coefficient).sum()
    control = _radial_factors(2, coefficient).sum()
    assert abs(value - control) > atol + rtol * abs(value), "the power must change the radial factor"


@pytest.mark.parametrize("power", [0, 1, 2, 3])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="One fixed two-electron configuration within 0.8 bohr of the ECP centre; no sampling or rejection.",
)
def test_native_nonlocal_value_radial_power_matches_closed_form_and_xla(power):
    """``jqmc_rs.compute_ecp_nonlocal_parts_all_pairs`` at power n equals the closed form and XLA.

    The native Sherman-Morrison non-local kernel (NN=1, Nv=12 icosahedral mesh,
    identity rotation). Comparison 1: the native value against the closed form
    (zone strict policy). Comparison 2: the native value against the same-mode
    XLA nearest-neighbour evaluator (XLA/Rust strict policy).
    """
    ham = _hamiltonian(power, _COEFFICIENT)
    r_up, r_dn, _ = _configuration(ham)
    wf = ham.wavefunction_data
    actual = float(
        jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(
            wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, r_up, r_dn, _MESH_W, _MESH_GP, False
        )
    )
    *_, xla = compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.eye(3), NN=1, Nv=12
    )
    xla = float(np.real(xla))
    reference = float(_radial_factors(power, _COEFFICIENT).sum())
    assert np.isfinite(actual) and np.isfinite(xla)

    atol, rtol = get_tolerance_min(_ZONES_CLOSED_FORM, "strict")
    _assert_power_is_live(power, _COEFFICIENT, atol, rtol)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=f"n={power}: closed form")
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    np.testing.assert_allclose(actual, xla, atol=atol, rtol=rtol, err_msg=f"n={power}: native vs XLA")


@pytest.mark.parametrize("power", [0, 1, 2, 3])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="One fixed two-electron configuration within 0.8 bohr of the ECP centre; no sampling or rejection.",
)
def test_native_vmc_local_energy_radial_power_matches_closed_form_and_xla(power, fast):
    """Native VMC local energy at power n carries the closed-form non-local energy and equals XLA.

    ``fast=True`` is ``jqmc_rs.compute_local_energy_fast`` (the production VMC
    energy with the geminal inverse), ``fast=False`` the full-recompute
    ``jqmc_rs.compute_local_energy``; Nv=12, identity rotation. Comparison 1: the
    non-local contribution e_L(c) - e_L(c=0) against the closed form (zone
    strict policy). Comparison 2: native e_L(c) against the same-mode XLA
    ``compute_local_energy_fast`` / ``compute_local_energy`` (XLA/Rust strict policy).
    """
    ham = _hamiltonian(power, _COEFFICIENT)
    off = _hamiltonian(power, 0.0)
    r_up, r_dn, g_inv = _configuration(ham)
    if fast:
        native = float(jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, _MESH_W, _MESH_GP, g_inv))
        native_off = float(jqmc_rs.compute_local_energy_fast(off, r_up, r_dn, _MESH_W, _MESH_GP, g_inv))
        xla = compute_local_energy_fast(ham, r_up, r_dn, jnp.eye(3), jnp.asarray(g_inv), None)
    else:
        native = float(jqmc_rs.compute_local_energy(ham, r_up, r_dn, _MESH_W, _MESH_GP))
        native_off = float(jqmc_rs.compute_local_energy(off, r_up, r_dn, _MESH_W, _MESH_GP))
        xla = compute_local_energy(ham, r_up, r_dn, jnp.eye(3))
    xla = float(np.real(xla))
    actual = native - native_off
    reference = float(_radial_factors(power, _COEFFICIENT).sum())
    assert np.isfinite(native) and np.isfinite(native_off) and np.isfinite(xla)

    atol, rtol = get_tolerance_min(_ZONES_CLOSED_FORM, "strict")
    _assert_power_is_live(power, _COEFFICIENT, atol, rtol)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=f"n={power}: closed form")
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    np.testing.assert_allclose(native, xla, atol=atol, rtol=rtol, err_msg=f"n={power}: native vs XLA")


@pytest.mark.parametrize("power", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="One fixed two-electron configuration within 0.8 bohr of the ECP centre; no sampling or rejection.",
)
def test_native_lrdmc_mesh_nonlocal_radial_power_matches_closed_form_and_xla(power, sign, non_local_move, fast):
    """Native LRDMC projection-step e_L and b_L at power n equal the closed form and XLA.

    ``jqmc_rs._gfmc_lrdmc_ecp`` is the native projection sub-step (alat=0.30,
    identity rotations, Nv=12, NN=1); ``fast=True`` is the production FAST
    mesh (Sherman-Morrison ECP T-move quadrature), ``fast=False`` the
    full-recompute mesh. Every mesh element of the l=0 term carries the sign of
    c: for c < 0 the whole non-local energy enters the fixed-node off-diagonal
    sum b_L, for c > 0 it enters the diagonal sign-flip term and b_L is
    unchanged. Comparisons 1 and 2: e_L(c) - e_L(c=0) against the closed form
    and b_L(c) - b_L(c=0) against its negative part (zone strict policy).
    Comparisons 3 and 4: native e_L and b_L against the production XLA
    ``_compute_V_elements_t`` (XLA/Rust projection policy, as in
    test_gfmc_projection.py).
    """
    coefficient = sign * _COEFFICIENT
    ham = _hamiltonian(power, coefficient)
    off = _hamiltonian(power, 0.0)
    r_up, r_dn, _ = _configuration(ham)
    dltmove = non_local_move == "dltmove"

    def native_step(h):
        wf = h.wavefunction_data
        e_L, b_L, _n = jqmc_rs._gfmc_lrdmc_ecp(
            wf.geminal_data,
            wf.jastrow_data,
            h.coulomb_potential_data,
            r_up,
            r_dn,
            _ALAT,
            np.eye(3),
            _MESH_W,
            _MESH_GP,
            1,
            dltmove,
            fast,
        )
        return float(e_L), float(b_L)

    e_on, b_on = native_step(ham)
    e_off, b_off = native_step(off)
    v_diag, v_nondiag, _ = _compute_V_elements_t(ham, r_up, r_dn, np.eye(3), non_local_move, _ALAT)
    xla_e = float(np.real(v_diag) + np.real(v_nondiag))
    xla_b = float(np.real(v_nondiag))
    factors = _radial_factors(power, coefficient)
    actual_e = e_on - e_off
    actual_b = b_on - b_off
    reference_e = float(factors.sum())
    reference_b = float(np.minimum(factors, 0.0).sum())
    assert np.all(np.isfinite([e_on, b_on, e_off, b_off, xla_e, xla_b]))

    atol, rtol = get_tolerance_min(_ZONES_CLOSED_FORM, "strict")
    _assert_power_is_live(power, coefficient, atol, rtol)
    np.testing.assert_allclose(actual_e, reference_e, atol=atol, rtol=rtol, err_msg=f"n={power}: e_L closed form")
    np.testing.assert_allclose(actual_b, reference_b, atol=atol, rtol=rtol, err_msg=f"n={power}: b_L closed form")
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GFMC, level="strict")
    np.testing.assert_allclose(e_on, xla_e, atol=atol, rtol=rtol, err_msg=f"n={power}: e_L native vs XLA")
    np.testing.assert_allclose(b_on, xla_b, atol=atol, rtol=rtol, err_msg=f"n={power}: b_L native vs XLA")

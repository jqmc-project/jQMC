"""Rust complex-twist HF stress of a B-spline-first pGTO(+)B-spline composite against strain AD of e_L.

The native complex B-spline kinetic stress (``complex_stress.rs``, ``bspline_kstress_side_complex``)
slices the B-spline rows out of the full AO-rank cotangent of a complex composite basis at
``bs0 = if gto_first { num_gto } else { 0 }``. It feeds the kinetic part of the VMC
Hellmann-Feynman stress ``jqmc_rs.vmc_grad_e_L_eps_pbc_complex`` and of the GFMC/LRDMC stress
``jqmc_rs._gfmc_stress_ecp_pbc_complex`` (through ``complex_kinetic_stress_w``). Every committed
complex composite fixture orders the pGTO child first, so the B-spline-first offset was never
reached: replacing it by ``bs0 = num_gto`` survived every test, including the PW-first test of
issue #47 (issue #58).

This module builds the committed ``pgto_bspline_twist_jnone`` cell (QE-projected Li ccECP at
k = 0.25, complex pGTO(+)B-spline composite, single determinant, no Jastrow) in two child orders,
reusing the reordering, configuration and GFMC strained-energy helpers of
``test_gfmc_complex_pw_first_stress`` (issue #47):

* ``gto_first``: the fixture as built (pGTO, B-spline);
* ``bspline_first``: children reordered to (B-spline, pGTO) and the MO coefficient columns permuted
  to match, so the wave function, and therefore every observable, is unchanged.

For each order, at one seeded electron configuration,

* the Rust VMC HF stress must equal the strain derivative of ``Re e_L``
  (``hamiltonians.compute_local_energy`` with the identity quadrature rotation), and
* the Rust GFMC stress must equal the strain derivative of the XLA continuum measurement energy
  (``jqmc_gfmc._compute_V_elements_t``),

both obtained by reverse-mode AD in full precision with the cell, nuclei and electrons
strained together (``Hamiltonian_data._apply_strain`` and ``r @ F.T``). The reference
differentiates the energies themselves and uses no Rust code and no analytic stress formula. In
full precision the VMC reference agrees with a fourth-order central finite difference of the same
energy to 1.1e-10 (recorded in issue #58); mixed precision has no finite-difference verdict, so each
mode is judged against AD in its own precision.

Precision scope (#163): the AD oracle below is full-only at strict. Mixed checks the same
native analytical stress tensors against independently rebuilt full native tensors from
identical serialized full physical input and deterministic coordinates. The central
parameter-block fidelity budget and an isolated mixed perturbation apply to every case.
The original full AD formulas, native arguments, cases and liveness thresholds remain.

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

import jqmc_rs
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from tests.test_rs_kernel._native_precision_fidelity import (
    full_source_hamiltonian_factory,
)
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.atomic_orbital import AOs_bspline_complex_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_complex_pw_first_stress import (
    _ALAT,
    _GRID,
    _NN,
    _WEIGHTS,
    _configuration,
    _pw_first_orb,
    _strained_energy,
)

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_PATTERN = "pgto_bspline_twist_jnone"
#: Non-local ECP mesh of the MCMC driver (``jqmc_mcmc.py``), also used by ``compute_local_energy``.
_VMC_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_VMC_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by the complex no-Jastrow VMC/GFMC local energy and its strain derivative.
#: Kept as a local literal so the GDD receipt validator can resolve the tolerance statically.
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")


def _hamiltonian(order):
    """Committed cell in its built (pGTO, B-spline) order or reordered to (B-spline, pGTO)."""
    ham = rs_ham(_PATTERN)
    geminal = ham.wavefunction_data.geminal_data
    for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        children = orb.aos_data.children
        assert len(children) == 2 and isinstance(children[1], AOs_bspline_complex_data), children
    if order == "bspline_first":
        # ``_pw_first_orb`` swaps the two children of any (pGTO, extra) leaf and permutes the MO columns.
        geminal = geminal.replace(
            orb_data_up_spin=_pw_first_orb(geminal.orb_data_up_spin),
            orb_data_dn_spin=_pw_first_orb(geminal.orb_data_dn_spin),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            assert isinstance(orb.aos_data.children[0], AOs_bspline_complex_data), orb.aos_data.children
    return ham


def _strained_local_energy(ham, r_up, r_dn):
    """Continuum VMC local energy at strain ``eps`` (cell, nuclei and electrons move; mesh fixed)."""

    def energy(eps):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        up, dn = jnp.asarray(r_up) @ deformation.T, jnp.asarray(r_dn) @ deformation.T
        return jnp.real(compute_local_energy(strained, up, dn, jnp.eye(3)))

    return energy


def _assert_nonvacuous(reference, atol, rtol):
    assert np.all(np.isfinite(reference)), reference
    # Every reference entry is distinguishable from zero at the comparison tolerance.
    assert np.all(np.abs(reference) > atol + rtol * np.abs(reference)), reference


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _native_derivative_outputs(ham, r_up, r_dn):
    """Original VMC and GFMC native HF measurement stress tensors."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    vmc = np.asarray(
        jqmc_rs.vmc_grad_e_L_eps_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _VMC_WEIGHTS, _VMC_GRID),
        dtype=np.float64,
    )
    gfmc = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_complex(
            geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False
        ),
        dtype=np.float64,
    )
    return {"VMC HF stress": vmc, "GFMC HF stress": gfmc}


@pytest.mark.parametrize("order", ["gto_first", "bspline_first"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_bspline_composite_hf_stress_matches_strain_ad_in_both_child_orders(order):
    """Rust complex VMC HF and GFMC stress equal strain AD of the energy for (pGTO, B-spline) and (B-spline, pGTO)."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            values = _native_derivative_outputs(fresh, *_configuration(fresh))
            expected_fields = {"GFMC HF stress", "VMC HF stress"}
            assert set(values) == expected_fields, values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                assert value.shape == (3, 3), (name, value.shape)
                assert np.all(np.isfinite(value)), (name, value)
                assert np.all(np.abs(value) > live_atol + live_rtol * np.abs(value)), (name, value)
            return values

        factory = full_source_hamiltonian_factory(lambda: _hamiltonian(order))
        mixed, full = mixed_vs_full(
            factory,
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ham = _hamiltonian(order)
    r_up, r_dn = _configuration(ham)
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    zero = jnp.zeros((3, 3))
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")

    values = _native_derivative_outputs(ham, r_up, r_dn)
    vmc = values["VMC HF stress"]
    vmc_reference = np.asarray(jax.jacrev(_strained_local_energy(ham, r_up, r_dn))(zero), dtype=np.float64)
    _assert_nonvacuous(vmc_reference, live_atol, live_rtol)
    assert np.all(np.isfinite(vmc)), vmc
    np.testing.assert_allclose(
        vmc,
        vmc_reference,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"Rust complex VMC HF stress != strain AD of Re e_L ({order})",
    )

    gfmc = values["GFMC HF stress"]
    gfmc_reference = np.asarray(jax.jacrev(_strained_energy(ham, r_up, r_dn))(zero), dtype=np.float64)
    _assert_nonvacuous(gfmc_reference, live_atol, live_rtol)
    assert np.all(np.isfinite(gfmc)), gfmc
    np.testing.assert_allclose(
        gfmc,
        gfmc_reference,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"Rust complex GFMC stress != strain AD of the measurement energy ({order})",
    )

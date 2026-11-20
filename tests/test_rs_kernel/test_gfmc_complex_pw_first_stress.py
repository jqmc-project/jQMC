"""Rust complex-twist GFMC stress on a PW-first pGTO(+)PW composite against strain AD of the energy.

``jqmc_rs._gfmc_stress_ecp_pbc_complex`` slices the GTO rows out of the full AO-rank
cotangent of a complex composite basis (``complex_stress.rs``, ``complex_kinetic_stress_w``:
``g0 = if gto_first { 0 } else { num_extra }``). Every committed complex composite fixture
orders the pGTO child before the plane-wave child, so the PW-first offset was never reached:
replacing it by ``g0 = 0`` survived every test (issue #47).

This module builds the committed ``pgto_pw_twist_jnone`` cell (QE-projected Li ccECP at
k = 0.25, complex pGTO(+)PW composite, single determinant, no Jastrow) in two child orders:

* ``gto_first``: the fixture as built (pGTO, PW);
* ``pw_first``: children reordered to (PW, pGTO) and the MO coefficient columns permuted
  to match, so the wave function, and therefore every observable, is unchanged.

For each order, the Rust stress at a seeded electron configuration must equal the strain
derivative of the XLA continuum measurement local energy
(``jqmc_gfmc._compute_V_elements_t``), obtained by reverse-mode AD in full precision
with the cell, nuclei and electrons strained together (``Hamiltonian_data._apply_strain``
and ``r @ F.T``) and the identity quadrature rotation held fixed. The reference differentiates
the energy itself and does not use any Rust slicing or analytic stress formula. In full
precision the same reference agrees with a fourth-order central finite difference of the
energy (recorded in the PR for issue #47); it is not repeated here because mixed precision has
no finite-difference verdict. The mixed numerical-fidelity scope is specified below.

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
from jqmc import jqmc_gfmc
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.atomic_orbital import AOs_composite_data, AOs_plane_wave_complex_data
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests.test_rs_kernel._rs_fixtures import rs_ham

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_PATTERN = "pgto_pw_twist_jnone"
_ALAT = 2.0
_SEED = 0
_WEIGHTS = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_GRID = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
_NN = 1
#: Zones crossed by the complex no-Jastrow measurement energy and its strain derivative.
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")


def _pw_first_orb(orb):
    """Reorder a (pGTO, PW) composite MO leaf to (PW, pGTO), permuting MO columns to match."""
    gto, pw = orb.aos_data.children
    num_gto = int(gto.num_ao)
    coefficients = jnp.asarray(orb.mo_coefficients)
    permuted = jnp.concatenate([coefficients[:, num_gto:], coefficients[:, :num_gto]], axis=1)
    return orb.replace(aos_data=AOs_composite_data.build((pw, gto)), mo_coefficients=permuted)


def _hamiltonian(order):
    ham = rs_ham(_PATTERN)
    geminal = ham.wavefunction_data.geminal_data
    for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        children = orb.aos_data.children
        assert len(children) == 2 and isinstance(children[1], AOs_plane_wave_complex_data), children
    if order == "pw_first":
        geminal = geminal.replace(
            orb_data_up_spin=_pw_first_orb(geminal.orb_data_up_spin),
            orb_data_dn_spin=_pw_first_orb(geminal.orb_data_dn_spin),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            assert isinstance(orb.aos_data.children[0], AOs_plane_wave_complex_data)
    return ham


def _configuration(ham):
    geminal = ham.wavefunction_data.geminal_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions)
    rng = np.random.default_rng(_SEED)
    r_up = positions[:num_up] + rng.normal(scale=0.5, size=(num_up, 3))
    r_dn = positions[:num_dn] + rng.normal(scale=0.5, size=(num_dn, 3))
    return np.asarray(r_up, dtype=np.float64), np.asarray(r_dn, dtype=np.float64)


def _strained_energy(ham, r_up, r_dn):
    """Continuum measurement energy at strain ``eps`` (cell, nuclei and electrons move; mesh fixed)."""

    def energy(eps):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        up, dn = jnp.asarray(r_up) @ deformation.T, jnp.asarray(r_dn) @ deformation.T
        return jnp.real(jqmc_gfmc._compute_V_elements_t(strained, up, dn, jnp.eye(3), "tmove", _ALAT)[2])

    return energy


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _native_derivative_outputs(ham, r_up, r_dn):
    """Original native GFMC measurement stress at the identical configuration."""
    wf = ham.wavefunction_data
    actual = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_complex(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            r_up,
            r_dn,
            _ALAT,
            np.eye(3),
            _WEIGHTS,
            _GRID,
            _NN,
            False,
        ),
        dtype=np.float64,
    )
    return {"GFMC HF stress": actual}


@pytest.mark.parametrize("order", ["gto_first", "pw_first"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_gfmc_complex_composite_stress_matches_strain_ad_in_both_child_orders(order):
    """Rust complex GFMC stress equals strain AD of the measurement energy for (pGTO, PW) and (PW, pGTO)."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            values = _native_derivative_outputs(fresh, *_configuration(fresh))
            expected_fields = {"GFMC HF stress"}
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
    actual = _native_derivative_outputs(ham, r_up, r_dn)["GFMC HF stress"]
    reference = np.asarray(jax.jacrev(_strained_energy(ham, r_up, r_dn))(jnp.zeros((3, 3))), dtype=np.float64)

    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    assert np.all(np.isfinite(reference)), reference
    assert np.all(np.isfinite(actual)), actual
    # Non-vacuous: every reference entry is distinguishable from zero at the comparison tolerance.
    assert np.all(np.abs(reference) > live_atol + live_rtol * np.abs(reference)), reference
    np.testing.assert_allclose(
        actual,
        reference,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"Rust complex composite stress != strain AD of the measurement energy ({order})",
    )

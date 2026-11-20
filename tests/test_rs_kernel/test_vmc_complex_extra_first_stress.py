"""Rust complex-twist VMC stress of extra-first pGTO composites against strain AD of ln|Psi| and e_L.

The native complex determinant Pulay stress (``complex_stress.rs``, ``pbc_det_pulay_stress_complex``)
slices the GTO rows out of the full AO-rank cotangent of a complex composite basis at
``g0 = if gto_first { 0 } else { num_extra }``. It feeds the VMC Pulay stress
``jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex`` directly, and the VMC Hellmann-Feynman stress
``jqmc_rs.vmc_grad_e_L_eps_pbc_complex`` through the complex ECP non-local mesh. Every committed
complex composite fixture orders the pGTO child first, so the extra-first offset was never
reached: replacing it by ``g0 = 0`` survived every test (issue #57).

This module builds the committed ``pgto_pw_twist_jnone`` and ``pgto_bspline_twist_jnone`` cells
(QE-projected Li ccECP at k = 0.25, complex composite, single determinant, no Jastrow) in two
child orders, reusing the reordering and configuration helpers of
``test_gfmc_complex_pw_first_stress`` (issue #47):

* ``gto_first``: the fixture as built (pGTO, extra);
* ``extra_first``: children reordered to (extra, pGTO) and the MO coefficient columns permuted to
  match, so the wave function, and therefore every observable, is unchanged.

For each case, the Rust Pulay stress must equal the strain derivative of ``Re ln Psi``
(``wavefunction.evaluate_ln_wavefunction``) and the Rust Hellmann-Feynman stress the strain
derivative of ``Re e_L`` (``hamiltonians.compute_local_energy`` with the identity quadrature
rotation), both obtained by reverse-mode AD in the active precision policy with the cell, nuclei and
electrons strained together (``Hamiltonian_data._apply_strain`` and ``r @ F.T``). The reference
differentiates the scalar functions themselves and uses no Rust code and no analytic stress
formula. In full precision it agrees with a fourth-order central finite difference of the same
functions to 1.1e-10 (recorded in issue #57); mixed precision has no finite-difference verdict, so
each mode is judged against AD in its own precision.

Precision scope (#163): the independent AD statements below apply in full at the strict budget.
Mixed evaluates only the native analytical quantities against the identical native route rebuilt
in full. Both policies load fresh Hamiltonians from the same full-precision serialized physical
input and use identical deterministic coordinates. Each homogeneous parameter block meets the
central fidelity budget and rejects an out-of-budget mixed perturbation. No native precision
zone is promoted and no mixed case is skipped.
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
from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_plane_wave_complex_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_complex_pw_first_stress import _configuration, _pw_first_orb


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Bind the central block norm directly to this module source."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = [pytest.mark.with_chdir]

#: Extra (non-GTO) child type of each committed complex composite fixture.
_EXTRA_CHILD = {
    "pgto_pw_twist_jnone": AOs_plane_wave_complex_data,
    "pgto_bspline_twist_jnone": AOs_bspline_complex_data,
}
#: Non-local ECP mesh of the MCMC driver (``jqmc_mcmc.py``), also used by ``compute_local_energy``.
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by ln|Psi| of a complex no-Jastrow determinant and its strain derivative.
_PULAY_ZONES = ("ao_eval", "ao_grad_lap", "det_eval")
#: Zones crossed by the complex no-Jastrow local energy and its strain derivative.
_HF_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")


def _hamiltonian(pattern, order):
    """Committed composite cell in its built (pGTO, extra) order or reordered to (extra, pGTO)."""
    ham = rs_ham(pattern)
    extra_type = _EXTRA_CHILD[pattern]
    geminal = ham.wavefunction_data.geminal_data
    for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        children = orb.aos_data.children
        assert len(children) == 2 and isinstance(children[1], extra_type), children
    if order == "extra_first":
        # ``_pw_first_orb`` swaps the two children of any (pGTO, extra) leaf and permutes the MO columns.
        geminal = geminal.replace(
            orb_data_up_spin=_pw_first_orb(geminal.orb_data_up_spin),
            orb_data_dn_spin=_pw_first_orb(geminal.orb_data_dn_spin),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            assert isinstance(orb.aos_data.children[0], extra_type), orb.aos_data.children
    return ham


def _strain_ad(ham, r_up, r_dn, scalar):
    """Reverse-mode AD of ``scalar(strained_ham, r_up F^T, r_dn F^T)`` with respect to ``eps`` at 0."""

    def at_strain(eps):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        return scalar(strained, jnp.asarray(r_up) @ deformation.T, jnp.asarray(r_dn) @ deformation.T)

    return np.asarray(jax.jacrev(at_strain)(jnp.zeros((3, 3))), dtype=np.float64)


def _assert_nonvacuous(reference, tolerance):
    atol, rtol = tolerance
    assert np.all(np.isfinite(reference)), reference
    # Every reference entry is distinguishable from zero at the comparison tolerance.
    assert np.all(np.abs(reference) > atol + rtol * np.abs(reference)), reference


def _native_derivative_outputs(ham, r_up, r_dn):
    """The original native analytical quantities, with identical kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    pulay = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(geminal, jastrow, r_up, r_dn), dtype=np.float64)
    hf = np.asarray(
        jqmc_rs.vmc_grad_e_L_eps_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID),
        dtype=np.float64,
    )
    return {"Pulay stress": pulay, "HF stress": hf}


@pytest.mark.parametrize("order", ["gto_first", "extra_first"])
@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("pgto_pw_twist_jnone", marks=pytest.mark.light),
        # About 40 s per row on the CPU reference host (B-spline strain AD), above the light budget.
        pytest.param("pgto_bspline_twist_jnone", marks=pytest.mark.heavy),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_vmc_complex_composite_stress_matches_strain_ad_in_both_child_orders(pattern, order):
    """Rust complex VMC Pulay and HF stress equal strain AD of ln|Psi| and e_L for both child orders."""
    if is_mixed_precision_enabled():
        entry_liveness = {
            "Pulay stress": get_tolerance_min(_PULAY_ZONES, "medium"),
            "HF stress": get_tolerance_min(_HF_ZONES, "medium"),
        }

        def analytical(fresh):
            values = _native_derivative_outputs(fresh, *_configuration(fresh))
            assert set(values) == set(entry_liveness), values.keys()
            for name, value in values.items():
                assert value.shape == (3, 3), (name, value.shape)
                # Keep the original per-entry predicate, using the mixed-entry
                # policy's medium threshold even during the full callback.
                _assert_nonvacuous(value, entry_liveness[name])
            return values

        factory = full_source_hamiltonian_factory(lambda: _hamiltonian(pattern, order))
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

    ham = _hamiltonian(pattern, order)
    r_up, r_dn = _configuration(ham)
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data

    values = _native_derivative_outputs(ham, r_up, r_dn)
    pulay = values["Pulay stress"]
    pulay_reference = _strain_ad(
        ham, r_up, r_dn, lambda h, up, dn: jnp.real(evaluate_ln_wavefunction(h.wavefunction_data, up, dn))
    )
    pulay_atol, pulay_rtol = get_tolerance_min(_PULAY_ZONES, "strict")
    _assert_nonvacuous(pulay_reference, get_tolerance_min(_PULAY_ZONES, "medium"))
    assert np.all(np.isfinite(pulay)), pulay
    np.testing.assert_allclose(
        pulay,
        pulay_reference,
        atol=pulay_atol,
        rtol=pulay_rtol,
        equal_nan=False,
        err_msg=f"Rust complex VMC Pulay stress != strain AD of Re ln Psi ({pattern}, {order})",
    )

    hf = values["HF stress"]
    hf_reference = _strain_ad(ham, r_up, r_dn, lambda h, up, dn: jnp.real(compute_local_energy(h, up, dn, jnp.eye(3))))
    hf_atol, hf_rtol = get_tolerance_min(_HF_ZONES, "strict")
    _assert_nonvacuous(hf_reference, get_tolerance_min(_HF_ZONES, "medium"))
    assert np.all(np.isfinite(hf)), hf
    np.testing.assert_allclose(
        hf,
        hf_reference,
        atol=hf_atol,
        rtol=hf_rtol,
        equal_nan=False,
        err_msg=f"Rust complex VMC HF stress != strain AD of Re e_L ({pattern}, {order})",
    )

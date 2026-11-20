"""Rust complex-twist kinetic MO derivative of an extra-first pGTO(+)PW / pGTO(+)B-spline composite against AD.

The native complex dT/dMO kernel stacks the raw AO blocks of a complex composite basis in
the child order (``complex_det.rs``, ``compute_aos_vgl_tuple``: the ``cat`` closure
``if orb.gto_first { vstack_c(a, b) } else { vstack_c(b, a) }``). Every committed complex
composite fixture orders the pGTO child first, so the extra-first branch was never reached:
replacing the condition by ``if true`` survived every test in full and mixed precision
(issue #76).

This module builds the committed ``pgto_pw_twist_jnone`` and ``pgto_bspline_twist_jnone``
cells (QE-projected Li ccECP at k = 0.25, complex composite, single determinant, no Jastrow)
in two child orders:

* ``gto_first``: the fixture as built (pGTO, extra);
* ``extra_first``: children reordered to (extra, pGTO) and the MO coefficient columns permuted
  to match, so the wave function, and therefore the kinetic energy, is unchanged.

For each order, at one seeded electron configuration, the native kinetic MO derivative
``d Re T / dC`` of both spin channels, from the free kernel ``jqmc_rs.grad_T_mo_withJ_complex``
and from the MCMC driver adapter ``_RsComplexPdCtx.pd_el_mo`` (the direct-MO kinetic block
of ``dE_L/dc`` for ``opt_geminal``), must equal reverse-mode AD of the XLA
``Re compute_kinetic_energy`` with respect to the geminal, read at
``orb_data_{up,dn}_spin.mo_coefficients`` (the JAX convention for a real function of a
complex leaf, ``dT/dRe C - i dT/dIm C``), in the active precision policy. The reference
differentiates the kinetic energy itself and uses no Rust code. In full precision the same
reference agrees with fourth-order central finite differences along Re and Im of the largest
entries (recorded in issue #76); it is not repeated here because mixed precision has no
finite-difference verdict.

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
from jqmc._precision import get_tolerance_min, precision_policy_snapshot, is_mixed_precision_enabled
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
)
from jqmc.jqmc_mcmc import _geminal_inv_batched, _RsComplexPdCtx
from jqmc.wavefunction import compute_kinetic_energy
from tests.test_rs_kernel._rs_fixtures import rs_ham


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Bind the central block norm directly to this module source."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

_EXTRA_TYPES = {
    "pgto_pw_twist_jnone": AOs_plane_wave_complex_data,
    "pgto_bspline_twist_jnone": AOs_bspline_complex_data,
}
_SEED = 1
_SCALE = 0.6
#: Zones crossed by the complex no-Jastrow kinetic energy and its MO-coefficient derivative.
_ZONES = ("ao_eval", "ao_grad_lap", "mo_eval", "mo_grad", "mo_lap", "det_eval", "det_grad_lap", "wf_kinetic")


def _extra_first_orb(orb):
    """Reorder a (pGTO, extra) composite MO leaf to (extra, pGTO), permuting MO columns to match."""
    gto, extra = orb.aos_data.children
    num_gto = int(gto.num_ao)
    coefficients = jnp.asarray(orb.mo_coefficients)
    permuted = jnp.concatenate([coefficients[:, num_gto:], coefficients[:, :num_gto]], axis=1)
    return orb.replace(aos_data=AOs_composite_data.build((extra, gto)), mo_coefficients=permuted)


def _hamiltonian(pattern, order):
    ham = rs_ham(pattern)
    extra_type = _EXTRA_TYPES[pattern]
    geminal = ham.wavefunction_data.geminal_data
    for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        children = orb.aos_data.children
        assert len(children) == 2 and isinstance(children[1], extra_type), children
        assert np.iscomplexobj(np.asarray(orb.mo_coefficients))
    if order == "extra_first":
        geminal = geminal.replace(
            orb_data_up_spin=_extra_first_orb(geminal.orb_data_up_spin),
            orb_data_dn_spin=_extra_first_orb(geminal.orb_data_dn_spin),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            assert isinstance(orb.aos_data.children[0], extra_type)
    return ham


def _configuration(ham):
    geminal = ham.wavefunction_data.geminal_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(_SEED)
    r_up = positions[np.arange(num_up) % len(positions)] + rng.normal(scale=_SCALE, size=(num_up, 3))
    r_dn = positions[np.arange(num_dn) % len(positions)] + rng.normal(scale=_SCALE, size=(num_dn, 3))
    return np.ascontiguousarray(r_up, dtype=np.float64), np.ascontiguousarray(r_dn, dtype=np.float64)


def _native_derivative_outputs(ham, r_up, r_dn):
    """The original native analytical quantities, with identical kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    geminal_inv = np.asarray(
        _geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1][0], dtype=np.complex128
    )
    free_up, free_dn = jqmc_rs.grad_T_mo_withJ_complex(geminal, jastrow, r_up, r_dn, geminal_inv)
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, True, True, True, precision_policy=precision_policy_snapshot())
    adapter_up, adapter_dn = _RsComplexPdCtx(geminal, jastrow, coulomb, ctx).pd_el_mo(r_up, r_dn, geminal_inv)

    return {
        "dT/dC up (free kernel)": np.asarray(free_up),
        "dT/dC dn (free kernel)": np.asarray(free_dn),
        "dT/dC up (driver adapter)": np.asarray(adapter_up),
        "dT/dC dn (driver adapter)": np.asarray(adapter_dn),
    }


@pytest.mark.light
@pytest.mark.parametrize("order", ["gto_first", "extra_first"])
@pytest.mark.parametrize("pattern", ["pgto_pw_twist_jnone", "pgto_bspline_twist_jnone"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_composite_kinetic_mo_derivative_matches_ad_in_both_child_orders(pattern, order):
    """Native complex dRe T/dC equals AD of the kinetic energy for (pGTO, extra) and (extra, pGTO)."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_derivative_outputs(fresh, r_up, r_dn)
            geminal = fresh.wavefunction_data.geminal_data
            up_shape = np.asarray(geminal.orb_data_up_spin.mo_coefficients).shape
            dn_shape = np.asarray(geminal.orb_data_dn_spin.mo_coefficients).shape
            expected_shapes = {
                "dT/dC up (free kernel)": up_shape,
                "dT/dC dn (free kernel)": dn_shape,
                "dT/dC up (driver adapter)": up_shape,
                "dT/dC dn (driver adapter)": dn_shape,
            }
            assert set(values) == set(expected_shapes), values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                assert value.shape == expected_shapes[name], (name, value.shape, expected_shapes[name])
                assert np.all(np.isfinite(value)), (name, value)
                assert np.max(np.abs(value)) > live_atol + live_rtol * np.max(np.abs(value)), (name, value)
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

    def kinetic(geminal_data):
        return jnp.real(compute_kinetic_energy(wf.replace(geminal_data=geminal_data), r_up, r_dn))

    grad = jax.grad(kinetic)(geminal)
    reference_up = np.asarray(grad.orb_data_up_spin.mo_coefficients, dtype=np.complex128)
    reference_dn = np.asarray(grad.orb_data_dn_spin.mo_coefficients, dtype=np.complex128)

    values = _native_derivative_outputs(ham, r_up, r_dn)

    checks = [
        ("dT/dC up (free kernel)", values["dT/dC up (free kernel)"], reference_up),
        ("dT/dC dn (free kernel)", values["dT/dC dn (free kernel)"], reference_dn),
        ("dT/dC up (driver adapter)", values["dT/dC up (driver adapter)"], reference_up),
        ("dT/dC dn (driver adapter)", values["dT/dC dn (driver adapter)"], reference_dn),
    ]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    for name, value, reference in checks:
        value = np.asarray(value, dtype=np.complex128)
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(reference)), (name, reference)
        assert np.all(np.isfinite(value)), (name, value)
        # Non-vacuous: the reference is distinguishable from zero at the comparison tolerance.
        assert np.max(np.abs(reference)) > live_atol + live_rtol * np.max(np.abs(reference)), (name, reference)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"Rust complex composite dT/dC != AD of the kinetic energy: {name} ({pattern}, {order})",
        )

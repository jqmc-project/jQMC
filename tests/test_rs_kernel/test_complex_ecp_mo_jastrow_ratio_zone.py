"""Issue #717: the complex-twist ECP dV_nl/dMO keeps its Jastrow ratio in the fp64 jastrow_ratio zone.

The precision policy (``jqmc/_precision.py``) assigns the ECP mesh Jastrow ratio
``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes); whole Jastrow
values belong to ``jastrow_eval`` (float32 in mixed). At a twist every other
input of the ECP non-local MO-coefficient derivative is float64 in mixed too:
complex periodic GTO values (``ao_bloch_eval``), the MO projection
(``mo_eval``), the determinant and its rank-one ratio (``det_eval`` /
``det_ratio``) and the ECP radial factors (``coulomb``). J1/J2 enter only
through the ratio, so in mixed this derivative must agree with an independently
rebuilt full-precision evaluation at the strict float64 budget of those zones.

The native kernel ``complex_stress::grad_v_ecp_nonlocal_mo_complex`` (reached
from the MCMC c-axis assembler through ``_RsComplexPdCtx.pd_el_ecp_mo``)
formed the ratio from two whole ``jastrow_value_complex`` values, which round
in ``jastrow_eval``. In mixed that left an fp32-sized error of up to 8e-6 in
the derivative (block scale 1e2) on the twist Li fixture below, while the XLA
route (``assemble._dV_ecp_dc``) was exact to the fp64 reference. On the
issue-717 input this native error was part of the per-walker dE_L/dc mismatch
that exceeded the strict mixed XLA--Rust pair on AMD EPYC.
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
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _MASK_DEFAULTS, _reduce_c_param_grads
from jqmc.jqmc_mcmc import _RsComplexPdCtx
from tests._precision_fidelity import mixed_vs_full
from tests.test_rs_kernel import test_batch_mesh_per_walker as per_walker
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_mcmc_observables import _geminal_inv, _is_complex, _walker_batch

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_PATTERN = "twist_sphepgto_j12"
_BLOCK = "mo_coefficients_up"
# Every zone on the twist ECP dV_nl/dMO path of a J1/J2 wavefunction; all are
# float64 in both modes.
_ZONES = ("ao_bloch_eval", "mo_eval", "det_eval", "det_ratio", "jastrow_ratio", "coulomb")


def _ecp_mo_derivatives(ham):
    """Native and XLA ECP dV_nl/dMO (``mo_coefficients_up`` block) for three rotated walkers."""
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    gem = wf.geminal_data
    assert _is_complex(ham) and bool(cp.ecp_flag)
    jas = wf.jastrow_data
    assert jas.jastrow_one_body_data is not None and jas.jastrow_two_body_data is not None
    assert jas.jastrow_three_body_data is None and jas.jastrow_dtn_data is None
    nw = per_walker._NW
    r_up, r_dn = _walker_batch(ham, nw=nw, seed=0)
    rts = per_walker._distinct_rotations(nw)
    ginv = np.stack([_geminal_inv(ham, r_up[w], r_dn[w]) for w in range(nw)]).astype(np.complex128)
    adapter = _RsComplexPdCtx(gem, jas, cp)
    assert bool(gem.time_reversal), "the tied mo_coefficients_up block is the fixture's MO source"
    native = []
    for w in range(nw):
        up, dn = adapter.pd_el_ecp_mo(r_up[w], r_dn[w], per_walker._W, per_walker._GP @ rts[w])
        native.append(np.asarray(up) + np.conj(np.asarray(dn)))
    xla_trees = [
        assemble._dV_ecp_dc(ham, jnp.asarray(r_up[w]), jnp.asarray(r_dn[w]), rts[w], geminal_inv=jnp.asarray(ginv[w]))
        for w in range(nw)
    ]
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xla_trees)
    xla = np.asarray(_reduce_c_param_grads(wf, stacked, nw, dict(_MASK_DEFAULTS))[_BLOCK])
    return {"r_up": np.asarray(r_up), "r_dn": np.asarray(r_dn), "native": np.stack(native), "xla": xla}


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("backend", ["native", "xla"])
def test_twist_ecp_mo_derivative_matches_full_precision_rebuild(backend):
    """Mixed twist ECP dV_nl/dMO equals the full rebuild at the strict float64 zone budget.

    ``twist_sphepgto_j12`` (lithium_k025.h5, bcc Li ccECP, k = (1/4, 1/4, 1/4),
    exp J1/J2), the three ``_walker_batch(seed=0)`` walkers and the three
    production rotations of ``test_batch_mesh_per_walker`` on the default Nv
    mesh. In mixed the actual value comes from a Hamiltonian loaded from the
    serialized full input under the mixed policy and the reference from an
    independently loaded full Hamiltonian; the electron configurations must be
    identical. In full both come from the same case (a consistency check).
    The derivative must be finite and live.
    """
    if is_mixed_precision_enabled():
        result, full = mixed_vs_full(
            full_source_hamiltonian_factory(lambda: rs_ham(_PATTERN)),
            _ecp_mo_derivatives,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
    else:
        result = full = _ecp_mo_derivatives(rs_ham(_PATTERN))
    np.testing.assert_array_equal(result["r_up"], full["r_up"])
    np.testing.assert_array_equal(result["r_dn"], full["r_dn"])
    actual = result[backend]
    reference = full["xla"]
    assert actual.shape == reference.shape == (per_walker._NW, 1, 26)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.max(np.abs(reference)) > 1.0, "inactive ECP MO derivative"
    # The fp64 native and XLA routes describe the same quantity (independent implementations).
    np.testing.assert_allclose(full["native"], full["xla"], atol=atol, rtol=rtol)
    for part, f in (("Re", np.real), ("Im", np.imag)):
        np.testing.assert_allclose(f(actual), f(reference), atol=atol, rtol=rtol, err_msg=f"{backend} ECP dV/dMO ({part})")

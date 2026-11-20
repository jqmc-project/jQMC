"""Native vs XLA parity: real-Gamma periodic Pulay stress with an active J3 and several electrons per spin.

The native periodic Pulay stress ``vmc_grad_ln_psi_eps_pbc`` adds the three-body
Jastrow value strain through ``stress_pbc::j3_pbc_value_strain``. Its J3 value
cotangent on the AO values carries a same-spin pair term built from the strict
lower-triangular pair matrix of each spin channel. With one electron per spin that
matrix is zero, so the pair term vanishes and a wrong index or transpose in it is
invisible. Every J3-bearing periodic stress parity row used the lithium / QE Li
fixtures with one up and one down electron, and the multi-electron periodic rows
(QE O-lsda) carried no J3, so such a fault survived every parity test (issue #854).

The input is the committed QE ``O-lsda_gamma`` save (real Gamma, triplet oxygen,
ccECP, 4 up / 2 dn electrons, independent up/dn orbitals) converted by the
production ``run_qe2jqmc`` (kind=pgto, J1 + J2), with an active pGTO J3 attached on
its AO leaf (the generated-matrix recipe ``_make_j3``). At fixed, well-conditioned
configurations the native ``vmc_grad_ln_psi_eps_pbc`` must equal the XLA analytic
``assemble.compute_grad_ln_Psi(wrt="eps")`` on the same geminal inverse, within the
strict Rust/XLA tier, in both precision modes.
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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc.derivatives import assemble  # noqa: E402
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from tests.test_jqmc_kernel._generated_matrix_base import _make_j3  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg  # noqa: E402
from tests.test_rs_kernel.test_periodic_lsda import _build_gamma  # noqa: E402

pytestmark = [pytest.mark.with_chdir]

_N_UP = 4
_N_DN = 2
_ZONES_PULAY_STRESS = ("ao_eval", "ao_grad_lap", "det_eval", "jastrow_eval", "jastrow_grad_lap")


def _with_jastrow(ham, jd):
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


@pytest.fixture(scope="module")
def ham_pair(tmp_path_factory):
    """(J1/J2-only, J1/J2 + active pGTO J3) QE O-lsda_gamma pgto hams."""
    ham_j12 = _build_gamma("pgto", tmp_path_factory.mktemp("o_lsda_gamma_j3"))
    gd = ham_j12.wavefunction_data.geminal_data
    assert (int(gd.num_electron_up), int(gd.num_electron_dn)) == (_N_UP, _N_DN)
    # Several electrons in EACH spin channel: the same-spin J3 pair term is live in both.
    assert min(_N_UP, _N_DN) >= 2 and _N_UP != _N_DN
    assert ham_j12.structure_data.pbc_flag, "the O-lsda_gamma fixture must be periodic"
    leaf = gd.orb_data_up_spin
    j3 = _make_j3(getattr(leaf, "aos_data", leaf))
    assert np.max(np.abs(np.asarray(j3.j_matrix)[:, :-1])) > 0.0, "the J3 pair block must be active"
    ham_j3 = _with_jastrow(ham_j12, ham_j12.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3))
    return ham_j12, ham_j3


def _xla_pulay_stress(ham, r_up, r_dn, g_inv):
    return np.asarray(
        assemble.compute_grad_ln_Psi(ham, jnp.asarray(r_up), jnp.asarray(r_dn), wrt="eps", geminal_inv=jnp.asarray(g_inv))
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [7, 3])
def test_pulay_stress_pbc_multi_electron_j3(ham_pair, seed):
    """Real-Gamma Pulay stress dln|Psi|/d(eps) with J3 on 4 up / 2 dn: native == XLA (strict)."""
    ham_j12, ham = ham_pair
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=seed, scale=0.6, cond_max=15.0)
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn), dtype=np.float64))

    reference = _xla_pulay_stress(ham, r_up, r_dn, g_inv)
    native = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, r_up, r_dn, g_inv))

    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_PULAY_STRESS, level="strict")
    assert reference.shape == (3, 3) and np.all(np.isfinite(reference))
    # Non-vacuity: the J3 changes the XLA Pulay stress by far more than the tolerance.
    j3_share = np.max(np.abs(reference - _xla_pulay_stress(ham_j12, r_up, r_dn, g_inv)))
    assert j3_share > 1.0e3 * (atol + rtol * np.max(np.abs(reference))), (seed, j3_share)
    np.testing.assert_allclose(
        native,
        reference,
        atol=atol,
        rtol=rtol,
        err_msg=f"seed={seed} periodic Pulay stress with J3, {_N_UP} up / {_N_DN} dn",
    )

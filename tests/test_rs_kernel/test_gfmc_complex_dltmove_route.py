"""The DLA flag must reach the complex GFMC PRODUCTION methods.

The complex ``run_gfmc_projection_{n,t}_complex`` PyO3 methods take a ``dltmove``
parameter; a method that dropped it (``dltmove: false`` internally) would silently run
the t-move Hamiltonian on every complex-twist ECP GFMC run that requested
``non_local_move="dltmove"`` -- both drivers.

Why the other tests cannot see that: ``test_gfmc_projection.py``'s
``test_complex_twist_chunk_honours_dltmove`` calls the FREE function
``jqmc_rs._gfmc_lrdmc_ecp_complex(..., dltmove)``, never the production method, so
the core can pass while the wrapper drops the flag; and
``test_multistep_projection_complex_twist_parity`` drives the production interval
but cannot select a non-local move, so it only exercises tmove. This module calls
the PRODUCTION methods with both settings.
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

import jqmc_rs
from jqmc._precision import precision_policy_snapshot

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import ATOL_BIT_EXACT, RTOL_BIT_EXACT

from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_forces import _pbc_cfg

_PATTERN = "twist_sphepgto_j12"
_ALAT = 0.30
_NN = 1


def _ctx_and_cfg():
    """Build the ctx exactly as production does (jqmc_gfmc.py:958)."""
    ham = rs_ham(_PATTERN)
    r_up, r_dn = _pbc_cfg(ham, 0)
    wf = ham.wavefunction_data
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        True,  # complex twist
        True,  # pbc
        False,  # want_stress
        want_force=False,
        want_gfmc=True,
        precision_policy=precision_policy_snapshot(),
    )
    return ctx, np.asarray(r_up, float), np.asarray(r_dn, float), ham


def _mesh(ham):
    from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default

    m = _NV_BASE_MESH[Nv_default]
    return np.asarray(m.weights, float), np.asarray(m.grid_points, float)


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
def test_production_complex_method_accepts_and_honours_dltmove(driver):
    """The production method must (a) accept ``dltmove`` and (b) produce a
    DIFFERENT trajectory than tmove -- identical output would mean the flag is
    dropped, which is exactly the defect this row guards."""
    ctx, r_up, r_dn, ham = _ctx_and_cfg()
    w, gp = _mesh(ham)
    k0, k1 = 11, 23

    def run(dlt: bool):
        if driver == "t":
            return ctx.run_gfmc_projection_t_complex(
                r_up, r_dn, _ALAT, 0.05, k0, k1, w, gp, random_mesh=False, nn=_NN, dltmove=dlt
            )
        # ``injected_rts`` empty (0,3,3) selects the FAST chain, mirroring the real
        # twin's own call sites; the reference-mesh (BITEXACT) chain is what a
        # non-empty stack would select.
        return ctx.run_gfmc_projection_n_complex(
            r_up, r_dn, _ALAT, 0.0, 1, k0, k1, np.zeros((0, 3, 3)), w, gp, random_mesh=False, nn=_NN, dltmove=dlt
        )

    tm = run(False)
    dl = run(True)

    e_tm = complex(tm["e_L"])
    e_dl = complex(dl["e_L"])
    assert np.isfinite(e_tm.real) and np.isfinite(e_dl.real), f"non-finite e_L: {e_tm}, {e_dl}"

    # The DLA Hamiltonian is not the t-move Hamiltonian: on this fixture the real
    # parts must differ. Equality here is the signature of the dropped flag.
    assert not np.isclose(e_tm.real, e_dl.real, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT), (
        f"driver={driver}: dltmove produced a bit-identical e_L to tmove "
        f"({e_tm.real!r} vs {e_dl.real!r}) -- the DLA flag is not reaching the complex kernel"
    )
    # Flag liveness is not backend parity. Re-evaluate each returned measurement
    # on its own endpoint using the corresponding XLA non-local convention;
    # never compare independently evolving tmove/dltmove trajectories.
    from jqmc._precision import get_tolerance_xla_rust_consistency
    from tests.test_rs_kernel.test_gfmc_projection import _xla_step_complex

    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio", "jastrow_eval", "jastrow_grad_lap", "ewald"),
        level="medium",
    )
    for move, result in (("tmove", tm), ("dltmove", dl)):
        reference = _xla_step_complex(
            ham, np.asarray(result["r_up"]), np.asarray(result["r_dn"]), _ALAT, np.eye(3), non_local_move=move
        )["e_L_meas"]
        np.testing.assert_allclose(complex(result["e_L"]), reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=move)

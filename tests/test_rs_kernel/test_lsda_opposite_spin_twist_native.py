"""Down-determinant conjugation leaves every real native fixed-phase quantity unchanged (Rust).

Native twin of ``tests/test_jqmc_kernel/test_lsda_opposite_spin_twist.py``
on the same converted LSDA pairs: ``ln|det|``, the real local energy, the
real non-local ECP energy and the real LRDMC local energy, branching energy
and candidate count of the native complex kernels agree between the
common-twist and the fully conjugated opposite-twist down determinant.
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

from jqmc._precision import get_tolerance_min
from jqmc._setting import NN_default, Nv_default
from jqmc.jqmc_mcmc import _NV_BASE_MESH
from tests._opposite_spin_twist_cases import KINDS, configurations, conjugate_partner, convert, rotation

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_ALAT = 0.30
_ZONES = (
    "ao_eval",
    "ao_bloch_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "jastrow_ratio",
    "det_eval",
    "det_grad_lap",
    "det_ratio",
    "wf_eval",
    "wf_kinetic",
    "wf_ratio",
    "coulomb",
    "ewald",
    "local_energy",
)


def _native(ham, up, dn, rt):
    mesh = _NV_BASE_MESH[Nv_default]
    w = np.asarray(mesh.weights, dtype=np.float64)
    gp = np.asarray(mesh.grid_points, dtype=np.float64) @ rt
    wf = ham.wavefunction_data
    gd, jd, cd = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    inverse = np.ascontiguousarray(np.linalg.inv(np.asarray(jqmc_rs.compute_geminal_complex(gd, up, dn))))
    out = {
        "ln_det": np.real(jqmc_rs.compute_ln_det_complex(gd, up, dn)),
        "e_L": jqmc_rs.compute_local_energy_fast_complex(ham, up, dn, w, gp, inverse)[0],
        "V_nonlocal": jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, jd, cd, up, dn, w, gp)[0],
    }
    for dlt in (False, True):
        e_re, _, b_re, _, n_cand = jqmc_rs._gfmc_lrdmc_ecp_complex(gd, jd, cd, up, dn, _ALAT, rt, w, gp, NN_default, dlt, True)
        out[f"lrdmc_{dlt}"] = np.array([e_re, b_re])
        out[f"candidates_{dlt}"] = n_cand
    return out


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lsda_twist_pair_native")
    return lambda kind: convert(kind, directory)


@pytest.mark.parametrize("kind", KINDS)
def test_native_down_conjugation_preserves_real_fixed_phase_terms(kind, converted):
    """Native ``ln|det|``, real e_L, real ECP energy and real LRDMC e_L / b_L / candidates agree.

    Both T-move and DLT-move LRDMC kernels run on the fast (carried-inverse)
    mesh route at the production lattice orientation.
    """
    ham = converted(kind)
    partner = conjugate_partner(ham)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    rt = rotation()
    for up, dn in configurations(ham):
        a, b = _native(ham, up, dn, rt), _native(partner, up, dn, rt)
        assert abs(a["V_nonlocal"]) > atol
        for key in a:
            if key.startswith("candidates_"):
                assert b[key] == a[key] > 0, key
            else:
                np.testing.assert_allclose(b[key], a[key], atol=atol, rtol=rtol, err_msg=key)

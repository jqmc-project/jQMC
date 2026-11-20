"""Bare-Z>1 periodic native sweeps must use the independent Kato image distance."""

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

import numpy as np
import pytest
import jqmc_rs

from jqmc._precision import precision_policy_snapshot
from jqmc.determinant import compute_geminal_all_elements
from tests._mcmc_kato_reference import DT, PATTERNS, assert_kato_replay, kato_case, replay

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("lattice_shift", [0, 30])
def test_native_sweep_kato_width_matches_independent_periodic_replay(pattern, lattice_shift, monkeypatch):
    """A native periodic Z_eff>1 chain matches the independent image-aware replay."""
    ham, charges, up, dn, key, normals = kato_case(pattern, lattice_shift)
    gem, jas, sd = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data, ham.structure_data
    matrix = np.asarray(compute_geminal_all_elements(gem, up, dn))
    kernel = jqmc_rs._mcmc_sweep_complex if np.iscomplexobj(matrix) else jqmc_rs._mcmc_sweep
    key_words = np.asarray(key, dtype=np.uint32)
    with jqmc_rs.NativeConstructionPrecision(precision_policy_snapshot()):
        actual = kernel(
            gem,
            jas,
            np.asarray(sd.positions, dtype=np.float64),
            charges,
            up,
            dn,
            matrix,
            np.linalg.inv(matrix),
            DT,
            0.0,
            int(key_words[0]),
            int(key_words[1]),
            normals,
            np.asarray(sd.cell, dtype=np.float64),
            int(sd._n_max),
        )
    expected = replay(ham, charges, up, dn, key, normals)
    actual["key"] = actual["final_key"]
    assert_kato_replay(actual, expected)
    if np.iscomplexobj(matrix):
        from tests.test_rs_kernel import test_complex_sweep as legacy

        monkeypatch.setattr(legacy, "_DT", DT)
        monkeypatch.setattr(legacy, "_NMOVES", len(normals))
        accepted = legacy._independent_acceptance_reference(
            gem, jas, np.asarray(sd.positions), charges, up, dn, normals, int(key_words[1]), np.asarray(sd.cell)
        )
        assert accepted == expected["accepted"]


@pytest.mark.parametrize("seed", [0, 2])
def test_complex_as_acceptance_oracle_uses_periodic_kato_distance(seed, monkeypatch):
    """The legacy AS-regularization oracle is periodic on the bare-Be twist cell."""
    from tests.test_rs_kernel import test_complex_sweep as legacy

    ham, charges, _, _, _, normals = kato_case("WF-PATTERN-016", 0)
    # The legacy AS oracle deliberately omits Jastrow ratios (its original
    # fixture is jnone); preserve that premise while changing the bare nucleus.
    from jqmc.jastrow_factor import Jastrow_data

    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=Jastrow_data()))
    gem, jas, sd = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data, ham.structure_data
    setup = (ham, gem, jas, np.asarray(sd.positions), charges, np.asarray(sd.cell), int(sd._n_max))
    monkeypatch.setattr(legacy, "_setup", lambda _pattern: setup)
    monkeypatch.setattr(legacy, "_DT", DT)
    monkeypatch.setattr(legacy, "_NMOVES", len(normals))
    with jqmc_rs.NativeConstructionPrecision(precision_policy_snapshot()):
        legacy.test_complex_sweep_as_regularization(seed)

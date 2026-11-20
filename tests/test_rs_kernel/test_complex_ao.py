"""Rust vs XLA: complex Bloch-pGTO AO values (Rust ``complex_ao.rs``) vs jqmc
``AOs_*_pgto_complex_data._compute_AOs`` at a twist k != Gamma (on the
committed complex-twist fixture). Foundation of the complex core (PBC twist).
Real Gaussian kernel + image-sum phase e^{ik.T_n}; evaluated at unwrapped r
(image-tiled) so no restore phase.
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

import jqmc_rs
import numpy as np
import pytest

from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from tests.test_rs_kernel._rs_fixtures import rs_ham

# Periodic (twist) AO evaluation carries the deliberate far-image AO cutoff, so
# the "medium" level applies.
_ZONES = ("ao_eval", "ao_grad_lap")

_PATTERN = "twist_sphepgto_jnone"  # lithium_k025.h5, ECP sphe-pGTO twist (k=0.25), SD, jastrow-none


def _orb_and_positions(side):
    ham = rs_ham(_PATTERN)
    gd = ham.wavefunction_data.geminal_data
    o = getattr(gd, side)
    aos = o.aos_data if type(o).__name__ == "MOs_data" else o
    assert "complex" in type(aos).__name__.lower(), type(aos).__name__
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    ne = gd.num_electron_up if "up" in side else gd.num_electron_dn
    return aos, R, int(ne)


@pytest.mark.light
@pytest.mark.parametrize("side", ["orb_data_up_spin", "orb_data_dn_spin"])
def test_compute_aos_complex(side):
    """Compute AOs complex."""
    aos, R, ne = _orb_and_positions(side)
    atol, rtol = _d8_backend_tolerance(("ao_eval",), level="strict")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        rng = np.random.default_rng(seed)
        r = R[np.arange(ne) % len(R)] + rng.normal(scale=0.7, size=(ne, 3))
        ref = np.asarray(aos._compute_AOs(r))
        got = np.asarray(jqmc_rs.compute_aos_complex(aos, r))
        d = np.abs(got - ref).max()
        assert d <= atol + rtol * np.abs(ref).max(), (seed, d)


@pytest.mark.light
@pytest.mark.parametrize("side", ["orb_data_up_spin", "orb_data_dn_spin"])
def test_compute_aos_vgl_complex(side):
    """Complex AO value + grad + Laplacian (the kinetic-energy inputs)."""
    aos, R, ne = _orb_and_positions(side)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    for seed in range(3):
        seed_label = str(seed)
        rng = np.random.default_rng(seed)
        r = R[np.arange(ne) % len(R)] + rng.normal(scale=0.7, size=(ne, 3))
        v, gx, gy, gz, lap = [np.asarray(a) for a in jqmc_rs.compute_aos_vgl_complex(aos, r)]
        ref_v = np.asarray(aos._compute_AOs(r))
        assert np.abs(v - ref_v).max() <= atol + rtol * np.abs(ref_v).max()
        grx, gry, grz = [np.asarray(a) for a in aos._compute_AOs_grad(r)]
        assert np.abs(gx - grx).max() <= atol + rtol * np.abs(grx).max()
        assert np.abs(gy - gry).max() <= atol + rtol * np.abs(gry).max()
        assert np.abs(gz - grz).max() <= atol + rtol * np.abs(grz).max()
        ref_lap = np.asarray(aos._compute_AOs_laplacian(r))
        assert np.abs(lap - ref_lap).max() <= atol + rtol * np.abs(ref_lap).max()

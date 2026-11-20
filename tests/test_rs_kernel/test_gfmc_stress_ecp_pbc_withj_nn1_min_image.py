"""LRDMC periodic ECP stress with Jastrow ranks the NN = 1 nucleus by minimum image (issue #212).

``gfmc_forces.rs::gfmc_stress_ecp_pbc_withj_route`` (the default LRDMC stress route of a real-Gamma
periodic ECP wavefunction with Jastrow, ``jqmc_rs._gfmc_stress_ecp_pbc_withj``; tmove and dltmove) ranks
the ECP nuclei for its ``NN = 1`` filter in the closure ``nearest_set`` by the minimum-image distance.
LRDMC walkers drift whole lattice vectors out of the home cell, so an electron reaches the ranking at a
position whose raw Cartesian nearest nucleus is not its minimum-image nearest one. A raw-distance ranking
there survives every other test (they place the electrons within the home cell).

Witness: the committed lithium cell (Li bcc, two ECP nuclei, real Gamma, SD + J1/J2), one up and one down
electron. The home-cell copy ``Xw`` puts each electron within about 1 bohr of a home-cell nucleus (raw ranking ==
minimum-image ranking); ``X = Xw + lattice vectors`` (four shift pairs) puts each electron next to an IMAGE
of that nucleus, where the raw Cartesian distance ranks the other nucleus first.

Independent references (each precision mode on its own; mixed uses the existing zone tolerances of its own
policy, no mixed-vs-full comparison): (1) lattice-translation invariance, ``Psi(r + L) = Psi(r)`` for a
real-Gamma wavefunction, so the HF stress ``de_L^t/d eps`` is invariant under moving an electron by a
lattice vector: kernel(X) == kernel(Xw); (2) the brute-force image search of
``tests/_ecp_min_image_reference.py`` establishes the input validity (the minimum-image nearest nucleus at X
is the one at Xw and differs from the raw nearest one for every electron).
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

import numpy as np
import pytest
import jqmc_rs

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests._ecp_min_image_reference import min_image_distances
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# Zones crossed by the periodic GFMC ECP stress (same set as the periodic force/stress tests).
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap", "ewald")
_ALAT = 0.30  # lattice spacing of the LRDMC move grid
# (lattice shift of the up electron, lattice shift of the down electron), in cell-vector units.
_SHIFTS = {
    "shift0": ((-1, 1, 1), (-3, 0, 0)),
    "shift1": ((-2, 1, 2), (-3, -1, 0)),
    "shift2": ((-1, 2, 1), (-3, 1, 1)),
    "shift3": ((-1, 1, 3), (-3, 0, 1)),
}


def _flat(x):
    if isinstance(x, (tuple, list)):
        return np.concatenate([_flat(y) for y in x])
    return np.asarray(x, dtype=np.float64).ravel()


def _configurations(ham, shift):
    """Home-cell copy ``Xw`` (each electron near its own nucleus) and ``X = Xw + lattice vectors``."""
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    pos = np.asarray(ham.structure_data.positions, dtype=np.float64)
    noise = np.random.default_rng(5).normal(scale=0.3, size=(2, 3))
    up_w = (pos[0] + noise[0])[None]
    dn_w = (pos[1] + noise[1])[None]
    shift_up, shift_dn = (np.asarray(s, dtype=np.float64) for s in _SHIFTS[shift])
    return (up_w + shift_up @ cell, dn_w + shift_dn @ cell), (up_w, dn_w)


def _assert_input_valid(ham, x, xw):
    """Brute-force image search: X sits next to an image of the Xw nucleus, which the raw distance ranks wrong."""
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    pos = np.asarray(ham.structure_data.positions, dtype=np.float64)
    r_x, r_w = np.vstack(x), np.vstack(xw)
    nearest_x = np.argmin(min_image_distances(pos, cell, r_x), axis=1)
    nearest_w = np.argmin(min_image_distances(pos, cell, r_w), axis=1)
    raw_x = np.argmin(np.linalg.norm(pos[None] - r_x[:, None], axis=-1), axis=1)
    raw_w = np.argmin(np.linalg.norm(pos[None] - r_w[:, None], axis=-1), axis=1)
    assert nearest_x.tolist() == nearest_w.tolist() == raw_w.tolist()
    assert np.all(raw_x != nearest_x), "the raw Cartesian distance must rank the wrong nucleus at X"


@pytest.mark.parametrize("shift", sorted(_SHIFTS))
@pytest.mark.parametrize("dltmove", [False, True], ids=["tmove", "dltmove"])
def test_gfmc_stress_ecp_pbc_withj_nn1_selects_min_image_nearest_nucleus(dltmove, shift):
    """The LRDMC periodic ECP stress with Jastrow (NN = 1) is lattice-translation invariant."""
    ham = rs_ham("pbc_gamma_sphepgto_j12")
    assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    x, xw = _configurations(ham, shift)
    _assert_input_valid(ham, x, xw)

    wf = ham.wavefunction_data
    gd, jd, cd = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)

    def stress(r):
        return _flat(jqmc_rs._gfmc_stress_ecp_pbc_withj(gd, jd, cd, r[0], r[1], _ALAT, np.eye(3), w, gp, 1, dltmove))

    s_x, s_w = stress(x), stress(xw)
    assert s_w.size == 9 and np.all(np.isfinite(s_x)) and np.all(np.isfinite(s_w))
    assert np.max(np.abs(s_w)) > 0.0, "the home-cell stress must be nonzero for the comparison to bind"
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    # Max-norm budget scaled by the largest stress entry (the precedent of the periodic GFMC stress tests in
    # test_gfmc_forces.py): the small off-diagonal entries carry the periodic far-image cutoff error of the
    # large diagonal ones.
    scale = max(1.0, float(np.max(np.abs(s_w))))
    max_diff = float(np.max(np.abs(s_x - s_w)))
    assert max_diff <= atol + rtol * scale, ("lattice translation", max_diff, atol + rtol * scale)

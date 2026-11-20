"""Native complex ECP dE_L/dc (MO coefficients) picks the minimum-image nearest nucleus (issue #172).

``complex_stress::grad_v_ecp_nonlocal_mo_complex`` (``const NN = 1``) is the
kernel behind ``jqmc_rs.grad_eL_ecp_wrt_mo_coeff_complex`` -- the ECP non-local
part of dE_L/dc that the native complex parameter-derivative driver
(``_NativeComplexObsCtx.pd_el_ecp_mo``) feeds to the linear method when MO
coefficients of a complex-twist ECP solid are optimized, on RAW (unwrapped)
walker coordinates -- and behind ``jqmc_rs.grad_V_ecp_nonlocal_wrt_mo_complex``.
Its ``nearest_set`` must rank the ECP nuclei by their minimum-image distance.

Witness: the committed lithium_k025 ccECP twist cell (Li bcc, two ECP nuclei,
k != Gamma), without Jastrow and with J1+J2, one up and one down electron. The
home-cell copy ``Xw`` puts each electron within about 1 bohr of a home-cell
nucleus; ``X = Xw + lattice vectors`` moves each electron next to an IMAGE of
that nucleus, where the raw Cartesian distance ranks the other nucleus first.

Independent references (each precision mode on its own, precision policy
of #163: mixed-vs-full is required only for analytical-vs-AD comparisons):
(1) lattice-translation invariance: for a Bloch twist wavefunction,
translating an electron by a lattice vector L multiplies Psi by
``exp(i k.L)``, a factor independent of the MO coefficients c, so E_L(c) and
every dE_L/dc are invariant -- the gradient at ``X`` equals the gradient at
``Xw``, where the raw and the minimum-image rankings agree; (2) the
brute-force image search of ``tests/_ecp_min_image_reference.py`` establishes
the input validity (the minimum-image nearest nucleus at ``X`` is the one at
``Xw`` and differs from the raw nearest one). In mixed precision the same
comparison runs mixed-vs-mixed at the existing zone tolerances of the active
policy.
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

from jqmc._precision import get_tolerance_min
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests._ecp_min_image_reference import min_image_distances
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# Zones of the native complex ECP dE_L/dc; local so the static tolerance
# resolver sees the exact budget.
_ZONES = (
    "ao_eval",
    "mo_eval",
    "jastrow_eval",
    "det_eval",
    "local_energy",
)
# (lattice shift of the up electron, lattice shift of the down electron), in cell-vector units.
_SHIFTS = {
    "shift0": ((1, 1, 0), (-1, 0, 0)),
    "shift1": ((1, 1, 1), (-1, -1, 0)),
    "shift2": ((-1, 1, 1), (0, -2, 1)),
    "shift3": ((2, -1, 3), (-3, 1, 0)),
}
_PATTERNS = ("twist_sphepgto_jnone", "twist_sphepgto_j12")


def _mesh():
    return (
        np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64),
        np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64),
    )


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


def _gradients(ham, x, xw):
    """MO-coefficient gradient blocks (up, dn) of both kernel bindings at X and at Xw."""
    w, gp = _mesh()
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    out = {}
    for tag, (up, dn) in (("X", x), ("Xw", xw)):
        e_l = jqmc_rs.grad_eL_ecp_wrt_mo_coeff_complex(gd, jd, cd, up, dn, w, gp)
        v_nl = jqmc_rs.grad_V_ecp_nonlocal_wrt_mo_complex(gd, cd, up, dn, w, gp)
        for name, blocks in (("grad_eL_ecp_wrt_mo_coeff", e_l), ("grad_V_ecp_nonlocal_wrt_mo", v_nl)):
            for spin, block in zip(("up", "dn"), blocks):
                out[(name, spin, tag)] = np.asarray(block)
    return out


def _assert_block_close(actual, expected, atol, rtol, label):
    """One logical gradient block: ``max|actual - expected| <= atol + rtol * max|expected|``."""
    assert actual.shape == expected.shape, label
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), label
    err = np.abs(actual - expected).max()
    bound = atol + rtol * np.abs(expected).max()
    assert err <= bound, f"{label}: max|diff| = {err:.3e} > {bound:.3e}"


@pytest.mark.parametrize("shift", sorted(_SHIFTS))
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_native_complex_ecp_mo_coeff_gradient_selects_min_image_nearest_nucleus(pattern, shift):
    """dE_L/dc of the complex ECP non-local term is invariant under lattice translation of the electrons."""
    base = rs_ham(pattern)
    assert base.structure_data.pbc_flag and base.coulomb_potential_data.ecp_flag
    x, xw = _configurations(base, shift)
    _assert_input_valid(base, x, xw)

    got = _gradients(base, x, xw)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for name, spin, tag in got:
        if tag != "X":
            continue
        signal = got[(name, spin, "Xw")]
        assert np.abs(signal).max() > 0.0, f"{name} {spin}: the gradient block carries no signal"
        _assert_block_close(got[(name, spin, "X")], signal, atol, rtol, f"lattice translation {name} {spin}")

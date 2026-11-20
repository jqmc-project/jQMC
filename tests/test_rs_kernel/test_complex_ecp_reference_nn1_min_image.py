"""Reference complex ECP fallback of the native VMC e_L picks the minimum-image nearest nucleus (issue #171).

``jqmc_rs.compute_local_energy_fast_complex`` evaluates the ECP non-local term
with the fast rank-1 kernel unless the fast state is unavailable; it then
falls back to the reference kernel ``complex_ecp::compute_ecp_nonlocal_withj_complex``
with ``nn = 1``, whose ``nearest_set`` ranks the ECP nuclei by their
minimum-image distance. The fallback runs when the debug/validation env
``JQMC_RS_ECP_COMPLEX_REFERENCE`` is set, or when the fast-state coverage gate
rejects a J3 ``j_matrix`` that is not in standard (C) layout, e.g. a
Fortran-ordered one. No other test reaches that ``nn = 1`` branch.

Witness: the committed lithium_k025 ccECP twist cell (Li bcc, two ECP nuclei,
k != Gamma), one up and one down electron. The home-cell copy ``Xw`` puts each
electron within about 1 bohr of a home-cell nucleus; ``X = Xw + lattice
vectors`` moves each electron next to an IMAGE of that nucleus, where the raw
Cartesian distance ranks the other nucleus first.

Independent references (each precision mode on its own, precision policy
of #163: mixed-vs-full is required only for analytical-vs-AD comparisons):
(1) lattice-translation invariance of E_L for a Bloch twist wavefunction
(``Psi(r + L) = exp(i k.L) Psi(r)``): the fallback e_L at ``X`` equals the
fallback e_L at ``Xw``, where the raw and the minimum-image rankings agree;
(2) the brute-force image search of ``tests/_ecp_min_image_reference.py``
establishes the input validity (the minimum-image nearest nucleus at ``X`` is
the one at ``Xw`` and differs from the raw nearest one); (3) the fallback e_L
at ``X`` equals the default fast route at ``X`` (the fast rank-1 selection is
pinned separately). In mixed precision the same comparisons run mixed-vs-mixed
at the existing zone tolerances of the active policy.
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
from jqmc.determinant import compute_geminal_all_elements
from tests._ecp_min_image_reference import min_image_distances
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_REFERENCE_ENV = "JQMC_RS_ECP_COMPLEX_REFERENCE"
# Zones of the native complex local energy; local so the static tolerance
# resolver sees the exact budget.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "jastrow_eval",
    "jastrow_grad_lap",
    "det_eval",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)
# (lattice shift of the up electron, lattice shift of the down electron), in cell-vector units.
_SHIFTS = {
    "shift0": ((1, 1, 0), (-1, 0, 0)),
    "shift1": ((1, 1, 1), (-1, -1, 0)),
    "shift2": ((-1, 1, 1), (0, -2, 1)),
    "shift3": ((2, -1, 3), (-3, 1, 0)),
}
# (fixture pattern, fallback trigger)
_ROUTES = {
    "jnone_env": ("twist_sphepgto_jnone", "env"),
    "j123_env": ("twist_sphepgto_j123", "env"),
    "j123_fortran_j_matrix": ("twist_sphepgto_j123", "fortran_j_matrix"),
}


def _mesh():
    return (
        np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64),
        np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64),
    )


def _fortran_j3(ham):
    """``ham`` with its J3 ``j_matrix`` stored in Fortran order (fails the fast-state layout gate)."""
    wf = ham.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    jm = np.asfortranarray(np.asarray(j3.j_matrix, dtype=np.float64))
    j3f = j3.replace(j_matrix=jm)
    jd = wf.jastrow_data.replace(jastrow_three_body_data=j3f)
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jd))


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


def _e_L(ham, up, dn):
    w, gp = _mesh()
    G = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up, dn), dtype=np.complex128)
    inv = np.ascontiguousarray(np.linalg.inv(G))
    return complex(*jqmc_rs.compute_local_energy_fast_complex(ham, up, dn, w, gp, inv))


def _energies(ham, trigger, x, xw, monkeypatch):
    """Fallback e_L at X and Xw (reference kernel, nn = 1) and the default fast e_L at X."""
    with monkeypatch.context() as m:
        m.delenv(_REFERENCE_ENV, raising=False)
        fast_x = _e_L(ham, *x)
        if trigger == "env":
            m.setenv(_REFERENCE_ENV, "1")
            fallback_ham = ham
        else:
            fallback_ham = _fortran_j3(ham)
            jm = fallback_ham.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix
            assert jm.flags.f_contiguous and not jm.flags.c_contiguous
        return {
            "fallback_X": _e_L(fallback_ham, *x),
            "fallback_Xw": _e_L(fallback_ham, *xw),
            "fast_X": fast_x,
        }


@pytest.mark.parametrize("shift", sorted(_SHIFTS))
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_native_complex_vmc_e_L_reference_ecp_fallback_selects_min_image_nearest_nucleus(route, shift, monkeypatch):
    """The nn = 1 reference ECP fallback e_L is lattice-translation invariant and equals the fast route."""
    pattern, trigger = _ROUTES[route]
    base = rs_ham(pattern)
    assert base.structure_data.pbc_flag and base.coulomb_potential_data.ecp_flag
    x, xw = _configurations(base, shift)
    _assert_input_valid(base, x, xw)

    got = _energies(base, trigger, x, xw, monkeypatch)
    for key, value in got.items():
        assert np.isfinite(value), f"{key} = {value} is not finite"
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    # Complex comparisons: |a - b| <= atol + rtol |b| on the complex modulus (Re and Im together).
    np.testing.assert_allclose(got["fallback_X"], got["fallback_Xw"], atol=atol, rtol=rtol, err_msg="lattice translation")
    np.testing.assert_allclose(got["fallback_X"], got["fast_X"], atol=atol, rtol=rtol, err_msg="fallback vs fast route")

"""Periodic real-Gamma ECP parameter derivatives rank the NN = 1 nucleus by minimum image (issue #211).

``param_grad_ecp.rs::collect_mesh`` serves the linear-method e_L parameter derivatives
``dE_L/dc`` (``grad_eL_ecp_wrt_mo_coeff``, ``grad_eL_ecp_wrt_lambda``,
``grad_eL_ecp_wrt_jastrow_2b_param``, ...). With the nearest-nucleus filter ``NN = 1`` its closure
``nearest_set`` ranks the ECP nuclei by the minimum-image distance. The MCMC walker carry is not
wrapped into the home cell for the ``e_L`` derivatives, so an electron that has drifted whole lattice
vectors out reaches the ranking at a position whose raw Cartesian nearest nucleus is not its
minimum-image nearest one. A raw-distance ranking there survives every other test (they place the
electrons within the home cell).

Witness: the committed lithium cell (Li bcc, two ECP nuclei, real Gamma; SD without Jastrow, SD + J1/J2,
and AGP), one up and one down electron. The home-cell copy ``Xw`` puts each electron within about 1 bohr of a
home-cell nucleus (raw ranking == minimum-image ranking); ``X = Xw + lattice vectors`` (four shift
pairs) puts each electron next to an IMAGE of that nucleus, where the raw Cartesian distance ranks the
other nucleus first.

Independent references (each precision mode on its own; mixed uses the existing zone tolerances of
its own policy, no mixed-vs-full comparison): (1) lattice-translation invariance, ``Psi(r + L) =
Psi(r)`` for a real-Gamma wavefunction, so every ``dE_L/dc`` is invariant under moving an electron by
a lattice vector: kernel(X) == kernel(Xw); (2) the brute-force image search of
``tests/_ecp_min_image_reference.py`` establishes the input validity (the minimum-image nearest nucleus
at X is the one at Xw and differs from the raw nearest one for every electron).

``test_param_grad_ecp_triclinic_lattice_translation`` (issue #569) applies reference (1) on two triclinic
cells (``tests/_triclinic_li_ecp.py``), where ``inv(cell) != inv(cell.T)`` and the orientation of the
cell inverse in the minimum-image fold becomes observable; every cubic route above is blind to it.
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
from tests._triclinic_li_ecp import TRICLINIC_CELL_MAPS, assert_cell_inverse_not_symmetric, triclinic_li_ecp_ham
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# Zones crossed by the ECP parameter derivatives (same set as the periodic ECP dE_L/dc tests).
_ZONES = ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval")
# (lattice shift of the up electron, lattice shift of the down electron), in cell-vector units.
_SHIFTS = {
    "shift0": ((-1, 1, 1), (-3, 0, 0)),
    "shift1": ((-2, 1, 2), (-3, -1, 0)),
    "shift2": ((-1, 2, 1), (-3, 1, 1)),
    "shift3": ((-1, 1, 3), (-3, 0, 1)),
}
# route -> (fixture, kernel name)
_ROUTES = {
    "sd_mo_jnone": ("pbc_gamma_sphepgto_jnone", "grad_eL_ecp_wrt_mo_coeff"),
    "sd_mo_j12": ("pbc_gamma_sphepgto_j12", "grad_eL_ecp_wrt_mo_coeff"),
    "sd_j2_j12": ("pbc_gamma_sphepgto_j12", "grad_eL_ecp_wrt_jastrow_2b_param"),
    "agp_lambda": ("lithium.h5", "grad_eL_ecp_wrt_lambda"),
}


def _ham(fixture):
    if fixture.endswith(".h5"):
        return _build_ham_for_cell(fixture, False, None, "AGP")
    return rs_ham(fixture)


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
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_param_grad_ecp_nn1_selects_min_image_nearest_nucleus(route, shift):
    """dE_L/dc of the periodic ECP kernel (NN = 1) is lattice-translation invariant."""
    fixture, kernel = _ROUTES[route]
    ham = _ham(fixture)
    assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    x, xw = _configurations(ham, shift)
    _assert_input_valid(ham, x, xw)

    wf = ham.wavefunction_data
    gd, jd, cd = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
    fn = getattr(jqmc_rs, kernel)
    g_x = _flat(fn(gd, jd, cd, x[0], x[1], w, gp))
    g_w = _flat(fn(gd, jd, cd, xw[0], xw[1], w, gp))

    assert g_w.size > 0 and np.all(np.isfinite(g_x)) and np.all(np.isfinite(g_w))
    assert np.max(np.abs(g_w)) > 0.0, "the home-cell derivative must be nonzero for the comparison to bind"
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    np.testing.assert_allclose(g_x, g_w, atol=atol, rtol=rtol, err_msg="lattice translation")


# Triclinic sibling (issue #569). Every route above runs on the cubic committed lithium cell,
# whose cell matrix is diagonal, so ``inv(cell) == inv(cell.T)`` there and a transposed cell
# inverse in ``collect_mesh`` leaves every displacement unchanged. On a triclinic cell the
# periodic minimum image of each electron-nucleus displacement is ``d - n @ cell`` with
# ``n = round(d @ inv(cell))``; any other matrix in place of ``inv(cell)`` stops the fold from
# being a lattice translation, so the derivative at ``X = Xw + lattice vectors`` no longer
# equals the derivative at ``Xw``. route -> native ECP non-local parameter-derivative kernel.
_TRICLINIC_ROUTES = {
    "mo_j12": "grad_eL_ecp_wrt_mo_coeff",
    "j1_j12": "grad_eL_ecp_wrt_jastrow_1b_param",
    "j2_j12": "grad_eL_ecp_wrt_jastrow_2b_param",
}


@pytest.mark.parametrize("shift", sorted(_SHIFTS))
@pytest.mark.parametrize("cell", sorted(TRICLINIC_CELL_MAPS))
@pytest.mark.parametrize("route", sorted(_TRICLINIC_ROUTES))
def test_param_grad_ecp_triclinic_lattice_translation(route, cell, shift):
    """dE_L/dc of the periodic ECP kernel is lattice-translation invariant on a triclinic cell.

    Triclinic Li ccECP J1+J2 Gamma Hamiltonian (``tests/_triclinic_li_ecp.py``): the home-cell
    copy ``Xw`` puts the up and the down electron at a Gaussian offset (scale 0.3 bohr) from the two nuclei, and
    ``X = Xw + lattice vectors`` (the four shift pairs above) moves them whole triclinic
    lattice vectors away. Reference: ``Psi(r + L) = Psi(r)`` for a real-Gamma wavefunction,
    so every ``dE_L/dc`` at ``X`` equals the one at ``Xw``. Full and mixed are each judged at
    the strict backend-consistency tier of their own zone policy (no mixed-vs-full comparison).
    """
    ham = triclinic_li_ecp_ham(cell)
    assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert_cell_inverse_not_symmetric(ham, atol, rtol)
    x, xw = _configurations(ham, shift)
    assert np.any(np.vstack(x) != np.vstack(xw))

    wf = ham.wavefunction_data
    gd, jd, cd = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    assert jd.jastrow_one_body_data is not None and jd.jastrow_two_body_data is not None
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
    fn = getattr(jqmc_rs, _TRICLINIC_ROUTES[route])
    g_x = _flat(fn(gd, jd, cd, x[0], x[1], w, gp))
    g_w = _flat(fn(gd, jd, cd, xw[0], xw[1], w, gp))

    assert g_w.size > 0 and np.all(np.isfinite(g_x)) and np.all(np.isfinite(g_w))
    assert np.max(np.abs(g_w)) > 0.0, "the home-cell derivative must be nonzero for the comparison to bind"
    np.testing.assert_allclose(g_x, g_w, atol=atol, rtol=rtol, err_msg=f"triclinic {cell} lattice translation")

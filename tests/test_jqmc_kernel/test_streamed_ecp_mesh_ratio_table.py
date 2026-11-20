"""Streamed ECP mesh derivatives use the determinant-ratio table on both sides.

The streamed (non-rank-one) ECP non-local derivatives weight per-configuration
fields by the mesh ratio ``det(G'_g) / det(G)``. That ratio comes from the
value-only ``compute_orb_api`` table (``ao_eval``, float32 in mixed). The
per-configuration fields receive Sherman-Morrison inverses from
``one_move_mesh.sm_geminal_inv_configs``. Because an inverse is supplied, they
read that same value-only table.

Without a running inverse, a base of those inverses taken from the fused
value table (``determinant._orbital_value_grad_lap_tables``, #681) would make
each ``(G'_g)^{-1}`` invert a geminal matrix different from the one whose
determinant is the weight. Near a node of ``G'_g`` the drift is
``~ 1/ratio``, so the fp32 table difference stops cancelling in
``ratio * O(G'_g)``.

In ``test_six_source_routes_energy_force_stress[True-dtn-average-qe-bspline]``
the tiled orbitals are always streamed; with a fused-table base the strain
derivative at [1, 2] is off fp64 by -1.6e-4, against a reported gap of
1.6455e-4, while the explicit supercell reference, which takes the rank-one
route, is off by 2.4e-6.

Two detectors:

* ``test_streamed_ecp_mesh_does_not_read_the_fused_value_table``: each
  streamed ECP derivative without an inverse is evaluated once normally and
  once with the fused value table replaced by NaN. The results must be
  identical. The no-inverse rank-one route reads the fused table (#681), so
  it must turn non-finite under the same patch; this shows the patch is live.
* ``test_tiled_streamed_ecp_strain_matches_supercell_rank_one_route``: on the
  frozen input of that node, the tiled ``d e_L / d eps`` (streamed ECP mesh)
  matches the explicit supercell route (rank-one ECP mesh). The tolerance is
  the source-route test's own elementwise budget.
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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import determinant
from jqmc.derivatives import assemble, ecp_force, gfmc, kinetic, one_move_mesh
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of
from tests.test_jqmc_kernel._generated_matrix_base import _apply_gem

pytestmark = [pytest.mark.without_chdir]

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


def _jastrow_hamiltonian(stem, gem):
    """The committed fixture's geminal with exponential J1 and J2 factors.

    The ``dltmove`` correction is ``P1 - P2``; it vanishes identically without a
    Jastrow ratio, so J1 (one-electron fixtures) and J2 keep those routes live.
    """
    structure, _, _, _, geminal, coulomb = read_trexio_file(trexio_file=os.path.join(_TREXIO, stem), store_tuple=True)
    jastrow = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure,
            core_electrons=core_electrons_of(coulomb, structure.natom),
            jastrow_1b_type="exp",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure
        ),
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    wavefunction = Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal)
    hamiltonian = Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)
    return _apply_gem(hamiltonian, gem)


def _configuration(hamiltonian, seed=0):
    geminal = hamiltonian.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(int(geminal.num_electron_up), 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(int(geminal.num_electron_dn), 3))),
    )


_RT = jnp.eye(3)


def _route(name):
    """One no-inverse ECP derivative entry point, as ``f(hamiltonian, r_up, r_dn)``."""
    return {
        "eps_pbc": lambda h, u, d: ecp_force.grad_V_ecp_wrt_strain(h, u, d, _RT),
        "eps_mol": lambda h, u, d: ecp_force.grad_V_ecp_wrt_strain_mol(h, u, d, _RT),
        "R": lambda h, u, d: ecp_force.grad_V_ecp_wrt_R(h, u, d, _RT),
        "r": lambda h, u, d: ecp_force.grad_V_ecp_wrt_r(h, u, d, _RT),
        "c": lambda h, u, d: assemble._dV_ecp_dc(h, u, d, _RT),
        "dltmove_r": lambda h, u, d: ecp_force.grad_V_ecp_nl_dltmove(h, u, d, _RT, wrt="r"),
        "dltmove_R": lambda h, u, d: ecp_force.grad_V_ecp_nl_dltmove(h, u, d, _RT, wrt="R"),
        "dltmove_eps_mol": lambda h, u, d: ecp_force.grad_V_ecp_nl_dltmove(h, u, d, _RT, wrt="eps"),
        "dltmove_eps_pbc": lambda h, u, d: gfmc._dltmove_delta_strain_pbc(h, u, d, _RT),
    }[name]


def _leaves(value):
    return np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree_util.tree_leaves(value)])


@pytest.mark.heavy
@pytest.mark.parametrize(
    "trexio_file,geminal,route",
    [
        ("lithium.h5", "SD", "eps_pbc"),
        ("lithium.h5", "SD", "R"),
        ("lithium.h5", "SD", "c"),
        ("lithium.h5", "SD", "dltmove_eps_pbc"),
        ("lithium_k025.h5", "SD", "eps_pbc"),
        ("lithium_k025.h5", "SD", "c"),
        ("Li_ecp_ccpvdz_cart.h5", "AGP", "eps_mol"),
        ("Li_ecp_ccpvdz_cart.h5", "SD", "r"),
        ("Li_ecp_ccpvdz_cart.h5", "AGP", "c"),
        ("N2_ecp_ccpvtz_cart.h5", "SD", "dltmove_r"),
        ("N2_ecp_ccpvtz_cart.h5", "SD", "dltmove_R"),
        ("N2_ecp_ccpvtz_cart.h5", "SD", "dltmove_eps_mol"),
    ],
)
def test_streamed_ecp_mesh_does_not_read_the_fused_value_table(trexio_file, geminal, route, monkeypatch):
    """A streamed no-inverse ECP derivative is independent of the fused orbital value table.

    The streamed route is selected by rejecting the rank-one fast path. Its mesh
    weights carry the value-only ratio, and every per-config field reads the
    value-only table, so the fused table must not enter. A fused-table value
    replaced by NaN must leave the result unchanged. Liveness: the rank-one
    no-inverse route builds its base from the fused table, so the same
    patch must make it non-finite.
    """
    hamiltonian = _jastrow_hamiltonian(trexio_file, geminal)
    r_up, r_dn = _configuration(hamiltonian)
    evaluate = _route(route)

    original = determinant._orbital_value_grad_lap_tables

    def poisoned(*args, **kwargs):
        orb_up, gou, lap_up, orb_dn, godn, lap_dn = original(*args, **kwargs)
        return jnp.full_like(orb_up, jnp.nan), gou, lap_up, jnp.full_like(orb_dn, jnp.nan), godn, lap_dn

    def poison():
        monkeypatch.setattr(determinant, "_orbital_value_grad_lap_tables", poisoned)
        monkeypatch.setattr(kinetic, "_orbital_value_grad_lap_tables", poisoned)

    # The patched module attribute is read at trace time, so each phase starts
    # from an empty compilation cache.
    if one_move_mesh.fast_supported(hamiltonian):
        # Liveness: the no-inverse rank-one base reads the fused table.
        with monkeypatch.context() as context:
            context.setattr(determinant, "_orbital_value_grad_lap_tables", poisoned)
            context.setattr(kinetic, "_orbital_value_grad_lap_tables", poisoned)
            jax.clear_caches()
            control = _leaves(evaluate(hamiltonian, r_up, r_dn))
        assert not np.all(np.isfinite(control)), "the rank-one no-inverse route must read the patched fused table"
    monkeypatch.setattr(one_move_mesh, "fast_supported", lambda _hamiltonian: False)
    jax.clear_caches()
    reference = _leaves(evaluate(hamiltonian, r_up, r_dn))
    poison()
    jax.clear_caches()
    probed = _leaves(evaluate(hamiltonian, r_up, r_dn))
    jax.clear_caches()
    assert np.all(np.isfinite(reference)) and np.any(reference != 0)
    np.testing.assert_array_equal(probed, reference)


@pytest.mark.heavy
def test_tiled_streamed_ecp_strain_matches_supercell_rank_one_route(tmp_path):
    """The tiled local-energy strain derivative matches the explicit supercell route.

    Input: the QE oxygen independent-spin B-spline twist average, tiling 2 x 1 x 1,
    with DTN Jastrow. This is the second sector (weight 0.8) of
    ``test_six_source_routes_energy_force_stress[True-dtn-average-qe-bspline]``,
    at the same configuration (``draw_well_conditioned_cfg``,
    seed 32, scale 0.6). Tiled orbitals always take the streamed ECP mesh;
    their explicit supercell expansion takes the rank-one route. Both
    describe the same wavefunction. The [1, 2] element of ``d e_L / d eps``
    is a near cancellation (about -0.0216 among components up to 168), so the
    ECP mesh error is visible there. The comparison uses the elementwise budget
    of ``test_six_source_routes_energy_force_stress``.
    """
    import tests.test_jqmc_kernel.test_orbital_tiling_source_routes as source_routes
    from jqmc._precision import get_tolerance_min
    from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

    hams, _ = source_routes._convert(tmp_path, "qe", "bspline", "average", True)
    tiled = source_routes._active_jastrow(hams[1], "dtn")
    supercell = source_routes._reference(tiled, "bspline")
    assert not one_move_mesh.fast_supported(tiled)
    assert one_move_mesh.fast_supported(supercell)
    r_up, r_dn = draw_well_conditioned_cfg(tiled, seed=32, scale=0.6)
    atol, rtol = get_tolerance_min(
        (
            "tiling_eval",
            "tiling_grad",
            "tiling_lap",
            "ao_eval",
            "ao_grad_lap",
            "det_grad_lap",
            "jastrow_grad_lap",
            "coulomb",
            "local_energy",
        ),
        "medium",
    )
    actual = np.asarray(assemble.compute_grad_e_L(tiled, r_up, r_dn, _RT, wrt="eps"))
    jax.clear_caches()
    desired = np.asarray(assemble.compute_grad_e_L(supercell, r_up, r_dn, _RT, wrt="eps"))
    jax.clear_caches()
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(desired))
    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol)

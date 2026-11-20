"""Deterministic per-hop checks of the carried electron-electron Ewald state."""

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

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import coulomb_potential as coulomb
from jqmc._precision import get_tolerance_min
from jqmc.structure import Structure_data
from jqmc.structure import _minimum_image_displacement

pytestmark = [pytest.mark.without_chdir]


@pytest.mark.light
@pytest.mark.parametrize(
    "cell",
    [
        pytest.param(((8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0)), id="cubic"),
        pytest.param(((8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 10.0)), id="tetragonal"),
        pytest.param(((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (0.0, 0.0, 10.0)), id="orthorhombic"),
        pytest.param(((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (2.0, 0.0, 10.0)), id="monoclinic"),
    ],
)
@pytest.mark.parametrize("n_up,n_dn", [(1, 0), (1, 1), (2, 2)])
def test_ewald_interval_matches_full_at_every_explicit_hop(cell, n_up, n_dn):
    """Carried Ewald energy equals full recomputation through 50 prescribed moves.

    The four cell shapes use explicit fractional electron coordinates, split
    into one, two, or four electrons. A cyclic electron index follows small
    (0.031, -0.027, 0.019) Bohr shifts; every tenth proposal translates by a
    lattice vector and every eleventh proposal is a no-op. Compare the initial
    state and every accepted update, without rebuilding the carried state.
    Explicit-index updates used by the driver must agree with the backwards
    compatible inferred-index call. A no-op preserves every component exactly.

    This is an ee Coulomb component test: no determinant, Jastrow, TREXIO input,
    cond(G), or stochastic sampler is involved. A Be ion supplies the setup
    structure; charged electron sectors retain the usual background term.
    Both precision modes use their existing strict Ewald-zone tolerance.
    The full Ewald kernel is an update-free reference, not an independent
    physical oracle for the common Ewald tables or minimum-image geometry.
    """
    cell = np.asarray(cell)
    structure = Structure_data.build(
        pbc_flag=True,
        positions=np.array([[0.0, 0.0, 0.0]]),
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
        atomic_numbers=(4,),
        element_symbols=("Be",),
        atomic_labels=("Be1",),
    )
    potential = coulomb.Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    ewald_data = coulomb.build_ewald_data(structure, potential)
    n_electron = n_up + n_dn
    fractional = np.array([[0.12, 0.23, 0.34], [0.41, 0.52, 0.63], [0.71, 0.82, 0.19], [0.91, 0.13, 0.47]])
    if n_electron > len(fractional):
        fractional = (np.arange(1, n_electron + 1)[:, None] * np.sqrt([2.0, 3.0, 5.0])[None, :]) % 1
    r_carts = jnp.asarray(fractional[:n_electron] @ cell)
    state = coulomb.compute_ewald_ee_interval_state(r_carts[:n_up], r_carts[n_up:], ewald_data)
    advance = jax.jit(coulomb.advance_ewald_ee_interval_state)
    evaluate = jax.jit(coulomb.compute_ewald_ee_from_interval_state, static_argnums=3)
    full_evaluation = jax.jit(coulomb.compute_ewald_potential_ee)
    atol, rtol = get_tolerance_min(("ewald",), "strict")
    for hop in range(51):
        hop_label = str(hop)
        actual = evaluate(*state, n_electron, ewald_data)
        reference = full_evaluation(r_carts[:n_up], r_carts[n_up:], ewald_data)
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"hop {hop_label}")
        if hop == 50:
            break
        index = hop % n_electron
        old_pos = r_carts[index]
        shift = jnp.asarray([0.031, -0.027, 0.019])
        if hop % 10 == 0:
            shift = jnp.asarray(cell[(hop // 10) % 3])
        if hop % 11 == 0:
            shift = jnp.zeros(3)
        new_pos = old_pos + shift
        inferred = advance(*state, r_carts, old_pos, new_pos, ewald_data)
        updated = advance(*state, r_carts, old_pos, new_pos, ewald_data, moved_index=index)
        for component, actual_component, reference_component in zip(
            ("real-space", "reciprocal-real", "reciprocal-imag"), updated, inferred, strict=True
        ):
            component_hop_label = f"{hop}/{component}"
            np.testing.assert_allclose(actual_component, reference_component, atol=atol, rtol=rtol, equal_nan=False)
        if hop % 11 == 0:
            for component, actual_component, reference_component in zip(
                ("real-space", "reciprocal-real", "reciprocal-imag"), updated, state, strict=True
            ):
                noop_label = f"{hop}/{component}"
                np.testing.assert_array_equal(actual_component, reference_component, err_msg=noop_label)
        state = updated
        r_carts = r_carts.at[index].set(new_pos)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "cell,n_up,n_dn",
    [
        (((8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0)), 64, 63),
        (((8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0)), 64, 64),
        (((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (2.0, 0.0, 10.0)), 64, 63),
        (((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (2.0, 0.0, 10.0)), 64, 64),
    ],
)
def test_ewald_interval_batch_boundary_preserves_every_hop(cell, n_up, n_dn):
    """Both sides of the 128-electron batching boundary preserve all 50 carried updates.

    Cubic and monoclinic cells contain 127 or 128 electrons at explicit
    fractional coordinates frac(i*(sqrt(2),sqrt(3),sqrt(5))), i=1..N.
    No RNG, conditioning filter, determinant or Jastrow is involved. Reuse the
    same prescribed moves, lattice translations, no-ops and unchanged central
    strict Ewald comparisons as the small-cell test. The two implementations
    must not require a carried-state refresh at the compile-time size boundary.
    """
    test_ewald_interval_matches_full_at_every_explicit_hop(cell, n_up, n_dn)


@pytest.mark.light
@pytest.mark.parametrize(
    "cell",
    [
        ((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (0.0, 0.0, 10.0)),
        ((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (2.0, 0.0, 10.0)),
        ((4.8, 6.4, 0.0), (-7.2, 5.4, 0.0), (0.0, 0.0, 10.0)),
    ],
)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_ewald_point_sum_shortcut_preserves_values_and_cell_jacobian(cell, sign):
    """Optimized point-image sums retain the general nearest-image value and cell Jacobian.

    Synthetic Ewald component, no wavefunction/artifact: orthorhombic,
    monoclinic and rotated-orthorhombic cells; fixed three-point/two-partner
    coordinates including either half-cell face, eta=0.2 and 27 explicit
    integer images. Cartesian points are held fixed while differentiating
    the cell, so off-diagonal cell perturbations cannot use a frozen
    orthogonality classification. Compare the unconditionally enumerated
    nearest-image formula, including its chosen derivative at ties, using
    the active strict Ewald policy in each precision mode. This verifies an
    optimization, not the physical correctness of the common Ewald formula.
    These deliberately finite image sums are not assumed converged: a wrong
    nearest image must remain visible rather than cancel through a nearly
    complete lattice sum.
    """
    cell = jnp.asarray(cell)
    r_carts = jnp.asarray([[sign * 0.5, 0.0, 0.0], [0.21, -0.43, 0.72], [0.49, 0.1, 0.49]]) @ cell
    partners = jnp.asarray([[0.0, 0.0, 0.0], [0.13, 0.25, 0.31]]) @ cell
    images = jnp.asarray(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))

    def general(lattice):
        inverse = jnp.linalg.inv(lattice)
        displacement = jax.vmap(jax.vmap(lambda vector: _minimum_image_displacement(vector, lattice, inverse, 2)))(
            r_carts[:, None, :] - partners[None, :, :]
        )
        squared = jnp.sum((displacement[:, :, None, :] + (images @ lattice)[None, None, :, :]) ** 2, axis=-1)
        distances = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
        return jnp.sum(jnp.where(squared > 0, jax.scipy.special.erfc(0.2 * distances) / distances, 0.0), axis=(1, 2))

    def optimized(lattice):
        return coulomb._ewald_ee_point_image_sum(
            r_carts, partners, jnp.asarray(0.2), images @ lattice, lattice, jnp.linalg.inv(lattice), 2
        )

    actual = np.asarray(optimized(cell))
    reference = np.asarray(general(cell))
    actual_jacobian = np.asarray(jax.jacfwd(optimized)(cell))
    reference_jacobian = np.asarray(jax.jacfwd(general)(cell))
    atol, rtol = get_tolerance_min(("ewald",), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_jacobian, reference_jacobian, atol=atol, rtol=rtol, equal_nan=False)

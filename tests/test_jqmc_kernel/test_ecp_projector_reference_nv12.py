"""Nv=12 and real-fixture nonlocal ECP references, independent of debug and ratio kernels."""

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

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_jnp, get_dtype_np, get_tolerance_min
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.determinant import compute_geminal_all_elements
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_wavefunction
from tests.test_jqmc_kernel.test_ecp_projector_reference import _gaussian_orbital_ratio, _projector_hamiltonian, _rotation


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"


def _quadrature_12():
    """Authored icosahedron vertices and equal weights, not read from the implementation's mesh.

    Two poles on the z axis and two staggered pentagons at z = +-1/sqrt(5) with
    in-plane radius 2/sqrt(5); the upper pentagon starts at azimuth 0 and the
    lower one at pi/5.
    """
    z = 1 / np.sqrt(5)
    rho = 2 / np.sqrt(5)
    upper = [[rho * np.cos(2 * k * np.pi / 5), rho * np.sin(2 * k * np.pi / 5), z] for k in range(5)]
    lower = [[rho * np.cos((2 * k + 1) * np.pi / 5), rho * np.sin((2 * k + 1) * np.pi / 5), -z] for k in range(5)]
    directions = np.asarray([[0, 0, 1], [0, 0, -1]] + upper + lower, dtype=float)
    return directions, np.full(12, 1 / 12)


@pytest.mark.parametrize("channel", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("position", ["first", "second"])
@pytest.mark.parametrize("rotation", ["identity", "tilted"])
@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="explicit", reason="Each named position is fixed; no sampling or conditioning rejection."
)
def test_each_ecp_channel_matches_independent_icosahedral_projection(channel, position, rotation):
    """Each nonzero l=0..6 channel matches an independent per-grid projection on the Nv=12 rule.

    Same synthetic molecular Hamiltonian as the Nv=18 channel reference: two
    pseudized H centres at (0,0,0) and (1.5,0.2,-0.1), one up electron/down
    empty, SD/MO of a normalized s Gaussian (alpha=0.8) on the second nucleus,
    no Jastrow. Positions first=(0.31,-0.27,0.49) or second=(-0.37,0.26,0.58);
    G is 1 by 1 with cond(G)=1 asserted. NN=1 selects the first nucleus, whose
    selected channel alone is nonzero. Nv=12 is the production default. The
    icosahedron coordinates/weights are authored here, Legendre values come from
    numpy, and the Gaussian ratio is closed-form. That ratio is
    evaluated where production evaluates it: orbital values in ao_eval (float32
    in mixed), the SD ratio in det_eval, and Legendre, weights and radial factor
    in coulomb (_gaussian_orbital_ratio). Per-grid values and the sum must
    agree under the existing Coulomb strict policy in full and mixed separately.
    This guarantees the discrete 12-point rule, not continuous integration.
    """
    ham = _projector_hamiltonian(0, 0, displaced=True, channel=channel)
    point = (0.31, -0.27, 0.49) if position == "first" else (-0.37, 0.26, 0.58)
    r_up = np.asarray([point])
    r_down = np.empty((0, 3))
    g = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_down))
    assert g.shape == (1, 1) and np.all(np.isfinite(g)) and g[0, 0] != 0
    cond_g = np.linalg.cond(g)
    assert cond_g == 1
    rotation_matrix = _rotation(rotation)
    result = compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data,
        ham.wavefunction_data,
        jnp.asarray(r_up),
        jnp.asarray(r_down),
        jnp.asarray(rotation_matrix),
        NN=1,
        Nv=12,
    )
    actual_grid = np.asarray(result[-2])
    actual_sum = np.asarray(result[-1])
    directions, weights = _quadrature_12()
    # The public RT argument rotates row vectors on the right.
    directions = directions @ rotation_matrix
    radius = np.linalg.norm(r_up[0])
    proposed = radius * directions
    centre = np.asarray([1.5, 0.2, -0.1])
    ratio = _gaussian_orbital_ratio(proposed, r_up[0], centre, 0, 0, 0.8, get_dtype_jnp("ao_eval"))
    cosine = directions @ (r_up[0] / radius)
    legendre = np.polynomial.legendre.legval(cosine, [0] * channel + [1])
    coefficient = (-1.0 if channel % 2 else 1.0) * (1 + channel / 5)
    radial = coefficient * np.exp(-(0.7 + channel / 10) * radius**2)
    reference_grid = np.asarray(radial * (2 * channel + 1) * weights * legendre * ratio, dtype=get_dtype_np("coulomb"))
    reference_sum = np.sum(reference_grid)
    assert actual_grid.shape == reference_grid.shape == (12,)
    assert np.all(np.isfinite(actual_grid)) and np.isfinite(actual_sum)
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    assert np.any(np.abs(reference_grid) > atol), "the selected ECP channel must be excited"
    np.testing.assert_allclose(actual_grid, reference_grid, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_sum, reference_sum, atol=atol, rtol=rtol, equal_nan=False)


def _raw_trexio_ecp(path):
    """Geometry and ECP tables read directly with h5py, bypassing the jqmc loader."""
    with h5py.File(path, "r") as handle:
        ecp = {key.removeprefix("ecp_"): np.asarray(handle["ecp"][key]) for key in handle["ecp"]}
        nuclei = np.asarray(handle["nucleus"]["nucleus_coord"], dtype=float)
        cell = None
        if "cell_a" in handle["cell"]:
            cell = np.asarray([handle["cell"][f"cell_{axis}"] for axis in "abc"], dtype=float)
    return nuclei, cell, ecp


def _nearest_images(point, nuclei, cell, count):
    """Brute-force nearest periodic image of every nucleus, then the ``count`` closest nuclei."""
    shifts = np.zeros((1, 3))
    if cell is not None:
        grid = np.arange(-2, 3)
        shifts = np.asarray(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T @ cell
    images = nuclei[:, None, :] + shifts[None, :, :]
    distances = np.linalg.norm(images - point, axis=-1)
    closest = images[np.arange(len(nuclei)), np.argmin(distances, axis=1)]
    order = np.argsort(np.linalg.norm(closest - point, axis=-1), kind="stable")[:count]
    return order, closest[order]


@jax.jit
def _dense_wavefunctions(wavefunction, up, down):
    return jax.vmap(evaluate_wavefunction, in_axes=(None, 0, 0))(wavefunction, up, down)


def _independent_nonlocal_reference(path, wavefunction, r_up, r_dn, rotation_matrix, nn):
    """Per-configuration nonlocal ECP values from raw TREXIO tables and dense Psi ratios.

    The ECP radial terms are c r^n exp(-a r^2) with TREXIO's own power n; the
    channel l = max_ang_mom_plus_1 of each nucleus is local and excluded. The
    authored icosahedron, numpy Legendre values, brute-force nearest images and
    many-electron Psi ratios from the dense ``evaluate_wavefunction`` supply
    every factor; no production mesh, projector or ratio kernel is used.
    Ordering follows the public result: up electrons, then down electrons, each
    over its NN nearest nuclei by increasing distance and then the 12 points.
    """
    nuclei, cell, ecp = _raw_trexio_ecp(path)
    directions, weights = _quadrature_12()
    directions = directions @ rotation_matrix
    configurations_up, configurations_dn, projectors, channel_rows = [], [], [], []
    translated_centres = 0
    for spin, electrons in ((0, r_up), (1, r_dn)):
        for index, point in enumerate(electrons):
            atoms, centres = _nearest_images(point, nuclei, cell, nn)
            for atom, centre in zip(atoms, centres, strict=True):
                translated_centres += int(not np.array_equal(centre, nuclei[atom]))
                radius = np.linalg.norm(point - centre)
                cosine = directions @ ((point - centre) / radius)
                projector = np.zeros(12)
                for ell in range(int(ecp["max_ang_mom_plus_1"][atom])):
                    terms = (ecp["nucleus_index"] == atom) & (ecp["ang_mom"] == ell)
                    radial = np.sum(
                        ecp["coefficient"][terms] * radius ** ecp["power"][terms] * np.exp(-ecp["exponent"][terms] * radius**2)
                    )
                    legendre = np.polynomial.legendre.legval(cosine, [0] * ell + [1])
                    channel = radial * (2 * ell + 1) * weights * legendre
                    projector = projector + channel
                    channel_rows.append((int(atom), ell, channel))
                for trial in centre + radius * directions:
                    up, dn = np.array(r_up), np.array(r_dn)
                    (up if spin == 0 else dn)[index] = trial
                    configurations_up.append(up)
                    configurations_dn.append(dn)
                projectors.append(projector)
    up = np.asarray(configurations_up)
    dn = np.asarray(configurations_dn)
    numerator = np.asarray(_dense_wavefunctions(wavefunction, jnp.asarray(up), jnp.asarray(dn)))
    denominator = np.asarray(evaluate_wavefunction(wavefunction, jnp.asarray(r_up), jnp.asarray(r_dn)))
    ratio = np.real(numerator / denominator)
    return (up, dn), np.concatenate(projectors) * ratio, (numerator, denominator), channel_rows, translated_centres


@pytest.mark.parametrize("trexio_file", ["silicon.h5", "CuBr_ecp_ccpvtz_cart.h5"])
@pytest.mark.parametrize("rotation", ["identity", "tilted"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_real_trexio_nonlocal_ecp_matches_independent_icosahedral_reference(trexio_file, rotation):
    """Real ccECP tables on silicon and CuBr match an independent Nv=12 projector reference.

    Committed TREXIO artifacts, loaded without a Jastrow: periodic Gamma silicon
    (8 Si, l=0,1 nonlocal, l=2 local, 16+16 valence electrons) and molecular
    CuBr (Cu l=0,1 nonlocal; Br l=0,1,2 nonlocal with l=3 local, 13+13
    electrons). One PCG64(seed=20260928) configuration: electrons are placed
    round-robin on the nuclei with normal(scale=0.5) offsets; no conditioning
    rejection, and every Psi is asserted finite and the reference Psi nonzero.
    NN=2, Nv=12 (production default), identity or the fixed tilted rotation.
    Silicon must select at least one translated (non-home) nearest image.

    The reference reads ECP coefficients/exponents/powers and geometry directly
    with h5py, authors the icosahedron, evaluates Legendre polynomials with
    numpy, chooses nearest periodic images by brute force, and multiplies
    by dense many-electron Psi ratios. Every proposed configuration, every
    per-grid value, and the sum agree under the existing Coulomb strict policy
    in full and mixed separately. The Br d nonlocal channel (CuBr) and a Si p
    channel (silicon) must contribute above that absolute tolerance. This
    guarantees the discrete projector on these fixtures, not continuous
    angular accuracy.
    """
    path = _TREXIO_DIR / trexio_file
    structure, _aos, _mos_up, _mos_dn, geminal, coulomb = read_trexio_file(str(path), store_tuple=True)
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
    nuclei = np.asarray(structure.positions, dtype=float)
    rng = np.random.default_rng(20260928)
    n_up, n_dn = geminal.num_electron_up, geminal.num_electron_dn
    r_up = nuclei[np.arange(n_up) % len(nuclei)] + rng.normal(scale=0.5, size=(n_up, 3))
    r_dn = nuclei[(np.arange(n_dn) + 1) % len(nuclei)] + rng.normal(scale=0.5, size=(n_dn, 3))
    rotation_matrix = _rotation(rotation)
    nn = 2
    result = compute_ecp_non_local_parts_nearest_neighbors(
        coulomb, wavefunction, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(rotation_matrix), NN=nn, Nv=12
    )
    reference = _independent_nonlocal_reference(path, wavefunction, r_up, r_dn, rotation_matrix, nn)
    (reference_up, reference_dn), reference_grid, (numerator, denominator), channel_rows, translated_centres = reference
    # Silicon must exercise at least one non-home nearest image; CuBr has none.
    assert (translated_centres > 0) == structure.pbc_flag
    actual_up, actual_dn = np.asarray(result[0]), np.asarray(result[1])
    actual_grid = np.asarray(result[-2])
    actual_sum = np.asarray(result[-1])
    count = (n_up + n_dn) * nn * 12
    assert actual_grid.shape == reference_grid.shape == (count,)
    assert np.all(np.isfinite(numerator)) and np.isfinite(denominator) and denominator != 0
    assert np.all(np.isfinite(reference_grid))
    assert np.all(np.isfinite(actual_grid)) and np.isfinite(actual_sum)
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    excited = {(atom, ell) for atom, ell, values in channel_rows if np.any(np.abs(values) > atol)}
    required = (1, 2) if trexio_file.startswith("CuBr") else (0, 1)
    assert required in excited, (required, excited)
    reference_sum = np.sum(reference_grid)
    np.testing.assert_allclose(actual_up, reference_up, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_dn, reference_dn, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_grid, reference_grid, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_sum, reference_sum, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()

"""MPC interpolation and table construction against independent constructions."""

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

from itertools import pairwise
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_complex_dtype_np, get_dtype_np
from jqmc._setting import (
    ATOL_MPC_OFFGRID_FIXTURE_REGRESSION,
    ATOL_MPC_OPEN_SHELL_FIXTURE_REGRESSION,
    ATOL_MPC_TABLE_FIXTURE_REGRESSION,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_TIGHT,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.atomic_orbital import AOs_bspline_data, AOs_plane_wave_complex_data, AOs_plane_wave_data
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.molecular_orbital import MOs_data
from jqmc.mpc import MPC_data, _periodic_cubic_bspline_coefficients, build_mpc_data, compute_mpc_potential_ee, evaluate_v_lr
from tests._opposite_spin_twist_cases import opposite_twist_partner
from tests.test_jqmc_kernel._mpc_independent_reference import (
    TABLE_FIXTURE_CUTOFF_HA,
    TABLE_FIXTURE_GRID,
    TABLE_FIXTURE_LENGTHS,
    TABLE_FIXTURE_MO_COEFFICIENTS,
    TABLE_FIXTURE_PW_MODES,
    TABLE_FIXTURE_TRANSFORMS,
    TABLE_FIXTURE_TWIST,
    box_kernel_coefficients,
    manufactured_field,
    periodic_cubic_reference,
    plane_wave_mpc_energy_reference,
    plane_wave_mpc_reference,
)
from tests.test_jqmc_kernel._noncubic_oracles import build_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CELLS = {
    "orthorhombic": np.diag([5.0, 6.0, 7.0]),
    "sheared-basis": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]) @ np.diag([5.0, 6.0, 7.0]),
    "triclinic": np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [-0.7, 1.2, 7.0]]),
}
_FRACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.13, 0.21, 0.37],
        [0.42, 0.53, 0.68],
        [0.81, 0.91, 0.17],
        [-0.21, 1.17, 2.31],
        [1.31, -0.43, 0.29],
    ]
)


def test_mpc_identity_mos_does_not_materialize_dense_coefficients(monkeypatch):
    """MPC consumes occupied-orbital leaves directly for implicit identity MOs."""
    structure = build_structure(np.diag([5.0, 6.0, 7.0]), np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]))
    aos = AOs_plane_wave_complex_data.build(structure, np.zeros((1, 3), dtype=np.float64), k_point=(0.0, 0.0, 0.0))
    mos = MOs_data.build_identity(aos_data=aos)
    sd = SD_data.build(num_electron_up=1, num_electron_dn=1, mos_up=mos)
    hamiltonian = SimpleNamespace(structure_data=structure, wavefunction_data=SimpleNamespace(geminal_data=sd))

    def _fail_if_materialized(_self):
        raise AssertionError("MPC materialized the implicit identity MO matrix")

    monkeypatch.setattr(MOs_data, "_mo_coefficients_jnp", property(_fail_if_materialized))
    mpc = build_mpc_data(hamiltonian, cutoff_ha=0.5, grid=(4, 4, 4))
    assert np.all(np.isfinite(np.asarray(mpc.v_lr_grid)))


@pytest.mark.parametrize("cell_name", list(_CELLS))
def test_periodic_cubic_lookup_matches_independent_spline(cell_name):
    """MPC's fp64 table lookup agrees with independent periodic cubic solves.

    Fixed Fourier-polynomial node values are input to both constructions.
    SciPy solves periodic cubic interpolation along each axis, while production
    uses FFT B-spline deconvolution and Cartesian-to-fractional lookup. The
    reference receives the known input fractions directly, so a wrong inverse
    cell, wrap, origin or interpolation coefficient cannot cancel between them.
    MPC table storage and evaluation are fp64 in both global precision modes;
    this is not a comparison of separately evaluated full/mixed calculations.
    """
    shape = (12, 14, 16)
    fractions = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), axis=-1)
    values = manufactured_field(fractions)
    cell = _CELLS[cell_name]
    origin = np.array([0.37, -0.41, 1.12])
    mpc = MPC_data(
        cell=jnp.asarray(cell, dtype=jnp.float64),
        origin=jnp.asarray(origin, dtype=jnp.float64),
        n_max=2,
        grid_shape=shape,
        v_lr_grid=jnp.asarray(values, dtype=jnp.float64),
        v_lr_coef=_periodic_cubic_bspline_coefficients(jnp.asarray(values, dtype=jnp.float64)),
        v_const=jnp.asarray(0.0, dtype=jnp.float64),
    )
    actual = np.asarray(evaluate_v_lr(mpc, jnp.asarray(origin + _FRACTIONS @ cell, dtype=jnp.float64)))
    reference = periodic_cubic_reference(values, _FRACTIONS)
    assert actual.dtype == reference.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert np.ptp(reference) > 0.0
    np.testing.assert_allclose(actual, reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)


def test_periodic_cubic_refinement_reduces_fixed_point_error():
    """Refining an MPC table reduces interpolation error against its analytic field.

    The fixed non-node fractions and Fourier polynomial are independent of
    production coefficients. Every grid refinement must reduce the maximum
    absolute error at these points; this is a convergence check, not an
    assertion of a newly chosen absolute energy-error allowance.
    """
    cell = _CELLS["triclinic"]
    origin = np.array([0.37, -0.41, 1.12])
    reference = manufactured_field(_FRACTIONS[1:])
    errors = []
    for size in (8, 12, 16, 24, 32):
        shape = (size,) * 3
        fractions = np.stack(np.meshgrid(*[np.arange(size) / size] * 3, indexing="ij"), axis=-1)
        values = jnp.asarray(manufactured_field(fractions), dtype=jnp.float64)
        mpc = MPC_data(
            cell=jnp.asarray(cell, dtype=jnp.float64),
            origin=jnp.asarray(origin, dtype=jnp.float64),
            n_max=2,
            grid_shape=shape,
            v_lr_grid=values,
            v_lr_coef=_periodic_cubic_bspline_coefficients(values),
            v_const=jnp.asarray(0.0, dtype=jnp.float64),
        )
        actual = np.asarray(evaluate_v_lr(mpc, jnp.asarray(origin + _FRACTIONS[1:] @ cell, dtype=jnp.float64)))
        assert np.all(np.isfinite(actual))
        errors.append(np.max(np.abs(actual - reference)))
    assert np.all(np.asarray(errors[:-1]) > np.asarray(errors[1:])), errors


def test_box_kernel_independent_quadrature_converges():
    """The independent box-kernel reference stabilizes at the existing fp64 budget.

    Three quadrature orders check the zero mode and three nonzero physical
    modes of the fixed orthorhombic cell. This checks the reference numerical
    integration, not the accuracy of a production MPC table.
    """
    lengths = np.array([5.0, 6.0, 7.0])
    modes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -1, 0]])
    vectors = 2.0 * np.pi * modes / lengths
    answers = [box_kernel_coefficients(lengths, vectors, order) for order in (20, 32, 48)]
    assert all(np.all(np.isfinite(answer)) for answer in answers)
    for previous, current in pairwise(answers):
        np.testing.assert_allclose(previous, current, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)


@pytest.mark.parametrize("basis_name", list(TABLE_FIXTURE_TRANSFORMS))
def test_production_table_within_fixture_regression_bound_of_independent_kernel(basis_name):
    """Production V_LR table and lookup stay within this fixture's regression bounds.

    ``build_mpc_data`` tabulates V_LR for a fixed doubly occupied plane-wave MO
    in the diag(5, 6, 7) Bohr box, in its orthorhombic and integer-shear basis
    representations (same physical Wigner-Seitz cell), cutoff 1.3 Ha, 16^3 grid.
    The reference (``plane_wave_mpc_reference``) integrates cos(G.r)/|r| over the
    box by face Gauss-Legendre quadrature and evaluates the density directly with
    NumPy; it uses none of the production Hood split, minimum-image scan, FFT
    extrapolation or B-spline lookup. The density phase is evaluated in the AO
    precision of the active mode, exactly where production's density is
    mode-dependent; the table, kernel and lookup are fp64 in both modes, so each
    mode is judged against its own reference and no full-mode result enters the
    mixed verdict.

    ATOL_MPC_TABLE_FIXTURE_REGRESSION / ATOL_MPC_OFFGRID_FIXTURE_REGRESSION are a
    regression bound for THIS fixture only (measured maxima rounded up), not a
    first-principles accuracy bound for other cells, densities or grids.
    """
    transform = TABLE_FIXTURE_TRANSFORMS[basis_name]
    cell = transform @ np.diag(TABLE_FIXTURE_LENGTHS)
    structure = build_structure(cell, np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]))
    aos = AOs_plane_wave_complex_data.build(
        structure, 2.0 * np.pi * TABLE_FIXTURE_PW_MODES / TABLE_FIXTURE_LENGTHS, k_point=tuple(TABLE_FIXTURE_TWIST)
    )
    mos = MOs_data.build(aos_data=aos, mo_coefficients=TABLE_FIXTURE_MO_COEFFICIENTS[None, :])
    sd = SD_data.build(
        num_electron_up=1, num_electron_dn=1, mos_up=mos, mos_dn=opposite_twist_partner(mos), time_reversal=False
    )
    hamiltonian = SimpleNamespace(structure_data=structure, wavefunction_data=SimpleNamespace(geminal_data=sd))
    size = TABLE_FIXTURE_GRID
    mpc = build_mpc_data(hamiltonian, cutoff_ha=TABLE_FIXTURE_CUTOFF_HA, grid=(size,) * 3)
    # The production grid origin is the nuclear centroid, which is zero here,
    # so table node (i, j, k) sits at fractions (i, j, k) / size.
    assert np.array_equal(np.asarray(mpc.origin), np.zeros(3))
    reference_nodes, reference_off_grid = plane_wave_mpc_reference(
        transform, get_dtype_np("ao_eval"), get_complex_dtype_np("ao_eval"), _FRACTIONS[1:]
    )
    table = np.asarray(mpc.v_lr_grid).reshape(-1)
    lookup = np.asarray(evaluate_v_lr(mpc, jnp.asarray(_FRACTIONS[1:] @ cell, dtype=jnp.float64)))
    assert table.dtype == lookup.dtype == reference_nodes.dtype == np.dtype(np.float64)
    assert table.shape == reference_nodes.shape and lookup.shape == reference_off_grid.shape
    assert np.all(np.isfinite(table)) and np.all(np.isfinite(reference_nodes))
    assert np.all(np.isfinite(lookup)) and np.all(np.isfinite(reference_off_grid))
    assert np.ptp(reference_nodes) > 0.0
    np.testing.assert_allclose(table, reference_nodes, atol=ATOL_MPC_TABLE_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(lookup, reference_off_grid, atol=ATOL_MPC_OFFGRID_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)


#: Open-shell fixture (issue #803): 3 up / 1 down electrons in the table-fixture box.
OPEN_SHELL_NUM_UP = 3
OPEN_SHELL_NUM_DN = 1
#: Integer plane-wave modes (physical box) of the four AO basis functions.
OPEN_SHELL_PW_MODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
#: Complex twist route: unitary 4-point DFT rows (entries i^(jk)/2) over the four
#: Bloch plane waves, so the four MOs are orthonormal over the box.
OPEN_SHELL_UNITARY = np.array([[1j ** (j * k) for k in range(4)] for j in range(4)]) / 2.0
#: Dense real MO route at Gamma: real AO functions cos(G1.r), sin(G2.r), cos(G3.r),
#: sin(G1.r) with G1, G2, G3 the [1,0,0], [0,1,0], [0,0,1] modes (mutually
#: orthogonal, equal norm), combined by OPEN_SHELL_ORTHOGONAL.
OPEN_SHELL_REAL_PW_MODES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
OPEN_SHELL_REAL_PW_IS_SIN = np.array([0.0, 1.0, 0.0, 1.0])
#: Implicit identity MO route at Gamma: each cubic B-spline AO is one orbital whose
#: values on the table nodes are OPEN_SHELL_ORTHOGONAL combinations of 1,
#: sqrt(2) cos(G3.r), sqrt(2) sin(G3.r), sqrt(2) cos(G2.r) (orthonormal on the nodes).
#: A single real plane wave cannot serve here: its |phi|^2 varies only at 2G,
#: outside the fixture cutoff, so every occupation pattern would give the same table.
OPEN_SHELL_SPLINE_PW_MODES = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]])
OPEN_SHELL_SPLINE_PW_IS_SIN = np.array([0.0, 0.0, 1.0, 0.0])
OPEN_SHELL_SPLINE_WEIGHTS = np.array([1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
#: Real orthogonal 4x4 Hadamard rows / 2 combining the real AO functions.
OPEN_SHELL_ORTHOGONAL = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float64) / 2.0
#: Electrons on table nodes (with whole-cell offsets): rows 0-2 up, row 3 down.
OPEN_SHELL_ELECTRON_NODES = np.array([[3, 5, 7], [11, 2, 13], [8, 15, 0], [1, 9, 4]])
OPEN_SHELL_ELECTRON_CELL_OFFSETS = np.array([[0, 0, 0], [-1, 1, 0], [2, 0, -1], [0, -3, 1]])
#: (geminal form, orbital route) of each open-shell case.
OPEN_SHELL_CASES = {
    "sd-mo-twist": ("sd", "mo-twist"),
    "agp-mo-twist": ("agp", "mo-twist"),
    "agpn-mo-twist": ("agpn", "mo-twist"),
    "sd-mo-gamma": ("sd", "mo-gamma"),
    "agp-mo-gamma": ("agp", "mo-gamma"),
    "agpn-mo-gamma": ("agpn", "mo-gamma"),
    "sd-identity-gamma": ("sd", "identity-gamma"),
    "agp-identity-gamma": ("agp", "identity-gamma"),
    "agpn-identity-gamma": ("agpn", "identity-gamma"),
}


def _open_shell_orbitals(route, cell, structure):
    """Production orbital container and an independent NumPy orbital evaluator for ``route``."""
    lengths = TABLE_FIXTURE_LENGTHS
    if route == "mo-twist":
        g_cart = 2.0 * np.pi * OPEN_SHELL_PW_MODES / lengths
        aos = AOs_plane_wave_complex_data.build(structure, g_cart, k_point=tuple(TABLE_FIXTURE_TWIST))
        mos = MOs_data.build(aos_data=aos, mo_coefficients=OPEN_SHELL_UNITARY)
        k_cart = TABLE_FIXTURE_TWIST @ (2.0 * np.pi * np.linalg.inv(cell).T)

        def evaluate(points, ao_real_dtype, ao_complex_dtype):
            phases = points.astype(ao_real_dtype) @ (g_cart + k_cart).astype(ao_real_dtype).T
            return np.exp(ao_complex_dtype(1j) * phases) @ OPEN_SHELL_UNITARY.T

        return mos, evaluate
    if route == "mo-gamma":
        g_cart = 2.0 * np.pi * OPEN_SHELL_REAL_PW_MODES / lengths
        aos = AOs_plane_wave_data.build(structure, g_cart, OPEN_SHELL_REAL_PW_IS_SIN)
        mos = MOs_data.build(aos_data=aos, mo_coefficients=OPEN_SHELL_ORTHOGONAL)

        def evaluate(points, ao_real_dtype, ao_complex_dtype):
            phases = points.astype(ao_real_dtype) @ g_cart.astype(ao_real_dtype).T
            return np.where(OPEN_SHELL_REAL_PW_IS_SIN > 0.5, np.sin(phases), np.cos(phases)) @ OPEN_SHELL_ORTHOGONAL.T

        return mos, evaluate

    g_cart = 2.0 * np.pi * OPEN_SHELL_SPLINE_PW_MODES / lengths

    def evaluate(points, ao_real_dtype, ao_complex_dtype):
        phases = points.astype(ao_real_dtype) @ g_cart.astype(ao_real_dtype).T
        basis = OPEN_SHELL_SPLINE_WEIGHTS * np.where(OPEN_SHELL_SPLINE_PW_IS_SIN > 0.5, np.sin(phases), np.cos(phases))
        return basis @ OPEN_SHELL_ORTHOGONAL.T

    # Fixture construction (not a reference): control points whose centred cubic
    # B-spline takes the prescribed orbital values on the 16^3 table nodes, by
    # dividing out the per-axis node stencil (1, 4, 1) / 6 in Fourier space.
    size = TABLE_FIXTURE_GRID
    nodes = np.array(np.meshgrid(*[np.arange(size) / size] * 3, indexing="ij")).reshape(3, -1).T
    values = evaluate(nodes @ cell, np.float64, np.complex128).T.reshape(4, size, size, size)
    stencil = (4.0 + 2.0 * np.cos(2.0 * np.pi * np.fft.fftfreq(size))) / 6.0
    symbol = stencil[:, None, None] * stencil[None, :, None] * stencil[None, None, :]
    control = np.fft.ifftn(np.fft.fftn(values, axes=(1, 2, 3)) / symbol, axes=(1, 2, 3)).real
    aos = AOs_bspline_data.build(structure, control, (size,) * 3, spline_order=3)
    return MOs_data.build_identity(aos_data=aos), evaluate


def _open_shell_geminal(form, mos):
    """Open-shell geminal whose occupied orbitals are MOs 0, 1, 2 (up) and MO 0 (down).

    The AGP / AGPn pairing is the Slater selector: paired diagonal (1, 0, 0, 0) and
    unpaired columns selecting MOs 1 and 2, so each form is the same determinant.
    """
    if form == "sd" and mos.identity:
        # Implicit identity SD MOs keep exactly the occupied rows: one leaf per spin.
        aos = mos.aos_data
        control = np.asarray(aos.spline_coefficients)
        mos_up, mos_dn = (
            MOs_data.build_identity(
                aos_data=AOs_bspline_data.build(aos.structure_data, control[:count], aos.spline_shape, spline_order=3)
            )
            for count in (OPEN_SHELL_NUM_UP, OPEN_SHELL_NUM_DN)
        )
        return SD_data.build(
            num_electron_up=OPEN_SHELL_NUM_UP,
            num_electron_dn=OPEN_SHELL_NUM_DN,
            mos_up=mos_up,
            mos_dn=mos_dn,
            time_reversal=False,
        )
    if form == "sd":
        return SD_data.build(
            num_electron_up=OPEN_SHELL_NUM_UP,
            num_electron_dn=OPEN_SHELL_NUM_DN,
            mos_up=mos,
            mos_dn=opposite_twist_partner(mos),
            time_reversal=False,
        )
    lambda_diag = np.array([1.0, 0.0, 0.0, 0.0])
    lambda_unpaired = np.zeros((4, OPEN_SHELL_NUM_UP - OPEN_SHELL_NUM_DN))
    lambda_unpaired[1, 0] = lambda_unpaired[2, 1] = 1.0
    if form == "agpn":
        return AGPn_data.build(
            num_electron_up=OPEN_SHELL_NUM_UP,
            num_electron_dn=OPEN_SHELL_NUM_DN,
            mos_up=mos,
            mos_dn=opposite_twist_partner(mos),
            lambda_diag=lambda_diag,
            lambda_unpaired=lambda_unpaired,
            time_reversal=False,
        )
    return AGP_data.build(
        num_electron_up=OPEN_SHELL_NUM_UP,
        num_electron_dn=OPEN_SHELL_NUM_DN,
        orb_data_up_spin=mos,
        orb_data_dn_spin=opposite_twist_partner(mos),
        # Complex orbital leaves require a complex128 pairing matrix.
        lambda_matrix=np.hstack([np.diag(lambda_diag), lambda_unpaired]).astype(
            np.result_type(mos.mo_coefficients, np.float64)
        ),
        time_reversal=False,
    )


@pytest.mark.parametrize("basis_name", list(TABLE_FIXTURE_TRANSFORMS))
@pytest.mark.parametrize("case_name", list(OPEN_SHELL_CASES))
@pytest.mark.gdd_electron_configuration(
    mode="required", reason="Fixed table-node electron coordinates for the open-shell MPC e-e energy check."
)
def test_open_shell_mpc_density_matches_occupied_orbital_reference(case_name, basis_name):
    """Open-shell MPC tables and energy use the density of every occupied orbital.

    Three up and one down electron occupy orthonormal four-plane-wave MOs in the
    table-fixture box (both basis representations), as an SD, as an AGP and as an
    AGPn whose unpaired columns carry two of the up electrons, through dense MO
    coefficients at the complex fixture twist, dense real MOs at Gamma, and implicit
    identity MOs at Gamma. The reference density is sum_{up} |psi_i|^2 +
    sum_{dn} |psi_j|^2, with each occupied orbital evaluated directly by NumPy on
    the table nodes (phase in the active AO precision), normalised to four
    electrons; ``plane_wave_mpc_reference`` / ``plane_wave_mpc_energy_reference``
    turn it into V_LR, V_const and the canonical MPC e-e energy of four electrons on
    table nodes with an exact orthorhombic minimum image. None of these share the
    production lambda-to-occupation reading, Hood split, FFT extrapolation or lookup.
    For dense MOs the natural-orbital occupations must also count the electrons of
    each spin (three up, one down). ATOL_MPC_OPEN_SHELL_FIXTURE_REGRESSION is a
    regression bound for this fixture only; each precision mode is judged against
    its own reference.
    """
    form, route = OPEN_SHELL_CASES[case_name]
    transform = TABLE_FIXTURE_TRANSFORMS[basis_name]
    cell = transform @ np.diag(TABLE_FIXTURE_LENGTHS)
    structure = build_structure(cell, np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]))
    mos, evaluate = _open_shell_orbitals(route, cell, structure)
    geminal = _open_shell_geminal(form, mos)
    hamiltonian = SimpleNamespace(structure_data=structure, wavefunction_data=SimpleNamespace(geminal_data=geminal))
    size = TABLE_FIXTURE_GRID
    mpc = build_mpc_data(hamiltonian, cutoff_ha=TABLE_FIXTURE_CUTOFF_HA, grid=(size,) * 3)
    assert np.array_equal(np.asarray(mpc.origin), np.zeros(3))

    if not mos.identity:
        _, _, occupation_up, _, _, occupation_dn = geminal.natural_orbital_density_coefficients_jnp()
        np.testing.assert_allclose(
            [float(np.sum(occupation_up)), float(np.sum(occupation_dn))],
            [OPEN_SHELL_NUM_UP, OPEN_SHELL_NUM_DN],
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    ao_real_dtype, ao_complex_dtype = get_dtype_np("ao_eval"), get_complex_dtype_np("ao_eval")
    nodes = np.array(np.meshgrid(*[np.arange(size) / size] * 3, indexing="ij")).reshape(3, -1).T
    orbitals = np.abs(evaluate(nodes @ cell, ao_real_dtype, ao_complex_dtype)).astype(np.float64) ** 2
    density = orbitals[:, :OPEN_SHELL_NUM_UP].sum(axis=1) + orbitals[:, :OPEN_SHELL_NUM_DN].sum(axis=1)
    assert np.ptp(density) > 0.0

    fractions = OPEN_SHELL_ELECTRON_NODES / size + OPEN_SHELL_ELECTRON_CELL_OFFSETS
    reference_nodes, reference_lookup = plane_wave_mpc_reference(
        transform,
        ao_real_dtype,
        ao_complex_dtype,
        fractions,
        node_density=density,
        n_electron=OPEN_SHELL_NUM_UP + OPEN_SHELL_NUM_DN,
    )
    table = np.asarray(mpc.v_lr_grid).reshape(-1)
    lookup = np.asarray(evaluate_v_lr(mpc, jnp.asarray(fractions @ cell, dtype=jnp.float64)))
    assert np.all(np.isfinite(table)) and np.all(np.isfinite(lookup))
    assert np.ptp(reference_nodes) > 0.0
    np.testing.assert_allclose(table, reference_nodes, atol=ATOL_MPC_OPEN_SHELL_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(
        lookup, reference_lookup, atol=ATOL_MPC_OPEN_SHELL_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO
    )

    r_all = fractions @ cell
    r_up, r_dn = r_all[:OPEN_SHELL_NUM_UP], r_all[OPEN_SHELL_NUM_UP:]
    v_mpc = float(compute_mpc_potential_ee(mpc, jnp.asarray(r_up), jnp.asarray(r_dn)))
    v_reference = plane_wave_mpc_energy_reference(transform, ao_real_dtype, ao_complex_dtype, r_up, r_dn, node_density=density)
    np.testing.assert_allclose(v_mpc, v_reference, atol=ATOL_MPC_OPEN_SHELL_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)

"""Autodiff oracles for complex-leaf AO derivatives and the complex twist kinetic energy.

Two oracle families:

1. AO level. For every complex AO leaf (Cartesian / spherical Bloch pGTO with
   lattice images and ``l <= 4``, complex plane wave, complex B-spline and a
   complex composite of all three) at the non-Gamma twist
   ``k = (0.25, -0.1, 0.4)``, and for real periodic pGTO leaves with
   ``l = 3`` and ``l = 4`` shells, the analytic ``compute_AOs_grad``,
   ``compute_AOs_laplacian`` and fused ``compute_AOs_value_grad_lap`` equal the
   forward-mode autodiff of ``compute_AOs`` itself. The complex value is
   differentiated with ``jax.jacfwd`` on the real electron coordinates, so the
   real and imaginary parts are differentiated separately and the full complex
   result is compared. Electrons straddle the home cell (some have
   ``floor(r @ inv(cell)) != 0``), so the pGTO wrap-restore phase and the
   B-spline / plane-wave ``e^{i k . r}`` factors are non-trivial.

2. Wavefunction level. On committed twist fixtures (``lithium_k025.h5`` spherical
   Bloch pGTO with ECP, ``H2_pbc_ae_k025_cart.h5`` Cartesian Bloch pGTO all
   electron), with and without J1 + J2 (no J3 at the twist), the analytic
   ``compute_kinetic_energy_all_elements`` and ``compute_kinetic_energy`` equal
   ``-1/2 (lap_i Psi) / Psi`` where the Laplacian is the autodiff Hessian trace
   of the complex ``Psi`` itself (``evaluate_wavefunction``), not of
   ``ln |Psi|``, so the phase derivatives are retained.

Negative controls replace one complex-leaf derivative method in-process
(``monkeypatch``, restored after the test) and require the same comparison to
fail for the intended reason: dropping the ``2 i k . grad u`` Bloch cross term
of the complex B-spline Laplacian, and applying the pGTO wrap-restore phase
twice to the Laplacian. Every verdict is deterministic (fixed seeds, no
sampling band); full and mixed each compare within their own strict zone
tolerances.
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

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._orbital_mixed_fidelity import assert_mixed_orbital_derivative_fidelity

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_complex_dtype_jnp, get_tolerance, get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_plane_wave_complex_data,
    _apply_image_expansion,
    _ComplexBlochPGTOMixin,
    _promote_pgto_to_complex,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
    is_complex_aos,
)
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data
from jqmc.wavefunction import compute_kinetic_energy, compute_kinetic_energy_all_elements, evaluate_wavefunction
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_jqmc_kernel.test_atomic_orbital import _make_aos_data

# Every Hamiltonian loads from an absolute committed-fixture path, so the
# autouse per-test cwd isolation is harmless.
pytestmark = [pytest.mark.with_chdir]

#: Non-Gamma fractional twist of every complex AO leaf built here.
_K_TWIST = (0.25, -0.1, 0.4)
#: Cubic cell edge for the plane-wave / B-spline / composite leaves (Bohr).
_A_SMALL = 6.0
_BSPLINE_SHAPE = (12, 12, 12)
_BSPLINE_ORDER = 5
#: Number of electrons at which each AO leaf is evaluated.
_N_ELECTRON = 5
#: A compared quantity must exceed its absolute tolerance by this factor
#: somewhere, otherwise the comparison would be vacuous.
_NON_VACUITY_FACTOR = 1.0e3

# Zones crossed by the AO comparisons: the AD side differentiates the
# ``ao_eval`` value, the analytic side lives in ``ao_grad_lap``.
_AO_ZONES = ("ao_eval", "ao_grad_lap")
# Zones crossed by the kinetic comparison: the AD side differentiates the
# ``wf_eval`` value (AO / MO / determinant / Jastrow forward zones), the
# analytic side assembles grad/lap zones into ``wf_kinetic``.
_KINETIC_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "wf_eval",
    "wf_kinetic",
)


# ---------------------------------------------------------------------------
# AO leaf builders.
# ---------------------------------------------------------------------------


def _small_cubic_structure() -> Structure_data:
    """Single-atom cubic periodic cell (edge ``_A_SMALL``), atom at the origin."""
    cell = np.eye(3, dtype=np.float64) * _A_SMALL
    return Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3), dtype=np.float64),
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )


def _pgto_leaf(ao_kind: str, l_max: int, *, complex_leaf: bool):
    """Periodic pGTO leaf with every shell ``l = 0..l_max`` on one off-centre atom.

    Cubic 10 Bohr cell, atom at ``(1, 1, 1)`` so the per-atom image truncation
    keeps several lattice images (non-trivial image fold). ``complex_leaf``
    promotes the same real leaf to the Bloch leaf at ``_K_TWIST``.
    """
    structure = make_pbc_structure(
        CELL_FIXTURES["cubic_10bohr"],
        positions=np.array([[1.0, 1.0, 1.0]], dtype=np.float64),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )
    base = _make_aos_data(structure, ao_kind, l_max=l_max, uniform=True)
    base.sanity_check()
    leaf = _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
    if complex_leaf:
        leaf = _promote_pgto_to_complex(leaf, _K_TWIST)
    leaf.sanity_check()
    return leaf


def _complex_pw_leaf(structure: Structure_data) -> AOs_plane_wave_complex_data:
    """Complex plane waves ``e^{i (G + k) . r}`` for a few +/- G (no Hermitian pairing)."""
    millers = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -1], [1, -1, 1], [-1, 1, 1]],
        dtype=np.float64,
    )
    return AOs_plane_wave_complex_data.build(structure, millers @ structure.recip_cell, k_point=_K_TWIST)


def _complex_bspline_leaf(structure: Structure_data) -> AOs_bspline_complex_data:
    """Interpolating complex B-spline of two smooth, genuinely complex periodic bands ``u_k``."""
    grids = [np.arange(n) / n for n in _BSPLINE_SHAPE]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    tp = 2.0 * np.pi
    b0 = np.sin(tp * fx) + 0.5 * np.cos(tp * fy) + 0.25 * np.sin(tp * fz)
    b1 = np.cos(tp * (fx + fy)) + 0.3 * np.sin(tp * fz)
    bands = np.stack([b0 + 0.5j * b1, b1 - 0.3j * b0 + 0.2], axis=0).astype(np.complex128)
    coeffs = bspline_prefilter_periodic(bands, _BSPLINE_ORDER)
    return AOs_bspline_complex_data.build(
        structure, coeffs.reshape(coeffs.shape[0], -1), _BSPLINE_SHAPE, _BSPLINE_ORDER, k_point=_K_TWIST
    )


def _complex_composite_leaf(structure: Structure_data):
    """Complex composite: Bloch spherical pGTO (+) complex plane wave (+) complex B-spline."""
    base = _make_aos_data(structure, "sphe", l_max=1, uniform=True)
    base.sanity_check()
    pgto = _promote_pgto_to_complex(_apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL), _K_TWIST)
    composite = pgto + _complex_pw_leaf(structure) + _complex_bspline_leaf(structure)
    composite.sanity_check()
    return composite


def _build_ao_case(case: str):
    """Return ``(aos_data, cell)`` for one AO case id."""
    if case.startswith("pgto_"):
        _, ao_kind, l_tag, kind = case.split("_")
        leaf = _pgto_leaf(ao_kind, int(l_tag[1:]), complex_leaf=(kind == "complex"))
        return leaf, np.asarray(CELL_FIXTURES["cubic_10bohr"], dtype=np.float64)
    structure = _small_cubic_structure()
    cell = np.asarray(structure.cell, dtype=np.float64)
    if case == "pw_complex":
        leaf = _complex_pw_leaf(structure)
    elif case == "bspline_complex":
        leaf = _complex_bspline_leaf(structure)
    elif case == "composite_complex":
        leaf = _complex_composite_leaf(structure)
    else:  # pragma: no cover - parametrisation guard
        raise ValueError(case)
    leaf.sanity_check()
    return leaf, cell


#: Im of the reference kinetic must exceed the allclose envelope by this factor somewhere.
_KINETIC_IM_MARGIN = 10.0
#: Integer cell offset of each electron: one in the home cell, the others in neighbouring cells.
_ELECTRON_CELL_OFFSETS = np.array([[0, 0, 0], [-1, 0, 0], [0, 1, 0], [1, -1, 1], [0, 0, -1]], dtype=np.float64)


def _straddling_electrons(cell: np.ndarray, seed: int) -> np.ndarray:
    """Electrons at fixed PCG64 fractional positions, placed in the home cell and its neighbours."""
    rng = np.random.default_rng(seed)
    frac = rng.uniform(0.05, 0.95, size=(_N_ELECTRON, 3)) + _ELECTRON_CELL_OFFSETS[:_N_ELECTRON]
    return (frac @ cell).astype(np.float64)


def _assert_straddles(r_carts: np.ndarray, cell: np.ndarray) -> None:
    """At least one electron outside the home cell and at least one inside."""
    n0 = np.floor(r_carts @ np.linalg.inv(cell))
    outside = np.any(n0 != 0, axis=1)
    assert outside.any(), "no electron outside the home cell: wrap-restore phase would be trivial"
    assert (~outside).any(), "no electron inside the home cell"


# Parametrisation: complex leaves at the twist + real/complex pGTO with l = 3, 4 shells.
_AO_CASES = (
    "pgto_cart_l4_real",
    "pgto_sphe_l4_real",
    "pgto_cart_l4_complex",
    "pgto_sphe_l4_complex",
    "pw_complex",
    "bspline_complex",
    "composite_complex",
)


def _ad_value_grad_lap(aos_data, r_carts):
    """Forward-mode AD of ``compute_AOs``: ``(value, (gx, gy, gz), lap)``.

    ``jax.jacfwd`` on the real flattened coordinates handles a complex output
    by differentiating its real and imaginary parts separately; the Laplacian
    of electron ``i`` is the trace of its own ``3 x 3`` Hessian block.
    """
    n_el = r_carts.shape[0]

    def f(x_flat):
        return compute_AOs(aos_data, x_flat.reshape(n_el, 3))

    x0 = jnp.asarray(r_carts, dtype=jnp.float64).reshape(-1)
    value = f(x0)
    jac = jax.jacfwd(f)(x0)  # (num_ao, N_e, 3 N_e)
    hess = jax.jacfwd(jax.jacfwd(f))(x0)  # (num_ao, N_e, 3 N_e, 3 N_e)
    idx = np.arange(n_el)
    grads = tuple(jac[:, idx, 3 * idx + d] for d in range(3))
    lap = sum(hess[:, idx, 3 * idx + d, 3 * idx + d] for d in range(3))
    return value, grads, lap


def _ao_fidelity_case(case):
    aos_data, cell = _build_ao_case(case)
    return aos_data, _straddling_electrons(cell, seed=20260927)


def _analytical_ao_outputs(case):
    aos_data, r_carts = case
    outputs = {
        "value": compute_AOs(aos_data, r_carts),
        "grads": compute_AOs_grad(aos_data, r_carts),
        "lap": compute_AOs_laplacian(aos_data, r_carts),
        "fused": compute_AOs_value_grad_lap(aos_data, r_carts),
    }
    # The fused output has exactly five fields.
    assert len(outputs["fused"]) == 5
    expected_shape = (aos_data.num_ao, r_carts.shape[0])
    for derivative in (*outputs["grads"], outputs["lap"], *outputs["fused"][1:]):
        assert np.shape(derivative) == expected_shape
    return outputs


def _compare_ao_fidelity(case, mixed, full):
    deviations = {}
    for name, actual, reference in (
        *((f"grad_{axis}", mixed["grads"][d], full["grads"][d]) for d, axis in enumerate("xyz")),
        ("lap", mixed["lap"], full["lap"]),
        *((f"fused_grad_{axis}", mixed["fused"][d + 1], full["fused"][d + 1]) for d, axis in enumerate("xyz")),
        ("fused_lap", mixed["fused"][4], full["fused"][4]),
    ):
        deviations[name] = assert_mixed_orbital_derivative_fidelity(actual, reference, f"{case} {name} mixed vs full")
    # Value API parity is a same-precision property, not a derivative fidelity check.
    val_atol, val_rtol = get_tolerance("ao_eval", "strict")
    np.testing.assert_allclose(
        np.asarray(mixed["fused"][0]),
        np.asarray(mixed["value"]),
        atol=val_atol,
        rtol=val_rtol,
        err_msg=f"{case} fused value vs compute_AOs",
    )
    return deviations


def _compare_ao_derivatives(case: str, aos_data, r_carts, *, full_reference=None) -> dict[str, float]:
    """Strict full AD or mixed analytical/full-analytical fidelity; retain value API parity."""
    if is_mixed_precision_enabled():
        if full_reference is None:
            mixed, full = mixed_vs_full(lambda: _ao_fidelity_case(case), _analytical_ao_outputs)
        else:
            mixed = jax.tree_util.tree_map(np.asarray, _analytical_ao_outputs((aos_data, r_carts)))
            full = full_reference
        return _compare_ao_fidelity(case, mixed, full)
    atol, rtol = get_tolerance_min(_AO_ZONES, "strict")
    val_u, grads_u, lap_u = _ad_value_grad_lap(aos_data, r_carts)
    grads_a = compute_AOs_grad(aos_data, r_carts)
    lap_a = compute_AOs_laplacian(aos_data, r_carts)
    val_f, gx_f, gy_f, gz_f, lap_f = compute_AOs_value_grad_lap(aos_data, r_carts)
    grads_f = (gx_f, gy_f, gz_f)
    deviations = {}
    for name, analytic, reference in (
        *((f"grad_{c}", grads_a[d], grads_u[d]) for d, c in enumerate("xyz")),
        ("lap", lap_a, lap_u),
        *((f"fused_grad_{c}", grads_f[d], grads_u[d]) for d, c in enumerate("xyz")),
        ("fused_lap", lap_f, lap_u),
    ):
        analytic = np.asarray(analytic)
        reference = np.asarray(reference)
        assert analytic.shape == reference.shape == (aos_data.num_ao, r_carts.shape[0]), name
        assert np.all(np.isfinite(analytic)) and np.all(np.isfinite(reference)), f"{case} {name}: non-finite"
        deviations[name] = float(np.max(np.abs(analytic - reference)))
        np.testing.assert_allclose(analytic, reference, atol=atol, rtol=rtol, err_msg=f"{case} {name} analytic vs AD")
    val_atol, val_rtol = get_tolerance("ao_eval", "strict")
    np.testing.assert_allclose(
        np.asarray(val_f), np.asarray(val_u), atol=val_atol, rtol=val_rtol, err_msg=f"{case} fused value vs compute_AOs"
    )
    return deviations


def _assert_non_vacuous(case: str, aos_data, r_carts) -> dict[str, float]:
    """Compared grads and Laplacians are far above the absolute tolerance (Re and Im for complex)."""
    atol, _ = get_tolerance_min(_AO_ZONES, "strict")
    floor = _NON_VACUITY_FACTOR * atol
    grads = compute_AOs_grad(aos_data, r_carts)
    lap = np.asarray(compute_AOs_laplacian(aos_data, r_carts))
    parts = {"grad": np.stack([np.asarray(g) for g in grads]), "lap": lap}
    for name, arr in parts.items():
        assert np.max(np.abs(arr.real)) > floor, f"{case} {name}: Re part vacuous"
        if is_complex_aos(aos_data):
            assert np.iscomplexobj(arr), f"{case} {name}: complex leaf returned a real array"
            assert np.max(np.abs(arr.imag)) > floor, f"{case} {name}: Im part vacuous"
    return {name: float(np.max(np.abs(arr))) for name, arr in parts.items()}


@pytest.mark.light
@pytest.mark.parametrize("case", _AO_CASES)
def test_complex_and_high_l_ao_derivatives_match_autodiff(case: str):
    """Analytic AO grad / Laplacian / fused outputs equal forward AD of the complex value.

    Covers each complex leaf family at ``k = (0.25, -0.1, 0.4)`` (Bloch pGTO cart
    and sphe with image fold and ``l <= 4``, complex plane wave, complex
    B-spline, complex composite) and real periodic pGTO with ``l = 3, 4``
    shells. Electrons straddle the home cell. Full retains its strict AD oracle;
    mixed uses the same analytical route rebuilt independently in full, with
    the central orbital-derivative fidelity budget. Value API parity stays same-mode.
    """
    aos_data, cell = _build_ao_case(case)
    r_carts = _straddling_electrons(cell, seed=20260927)
    _assert_straddles(r_carts, cell)
    if case.startswith("pgto_"):
        assert max(aos_data.angular_momentums) == 4, "high-l row must carry l = 4 shells"
        assert 3 in set(aos_data.angular_momentums), "high-l row must carry l = 3 shells"
        assert aos_data._n_image > 1, "periodic pGTO row must fold more than one lattice image"
    magnitudes = _assert_non_vacuous(case, aos_data, r_carts)
    deviations = _compare_ao_derivatives(case, aos_data, r_carts)
    print(f"{case}: max |analytic| = {magnitudes}, max derivative deviation = {deviations}")
    jax.clear_caches()


def _bspline_laplacian_without_cross_term(self, r_carts):
    """Mutant: complex B-spline Laplacian with the ``2 i k . grad u`` term dropped."""
    zone = "ao_grad_lap"
    u, _ux, _uy, _uz, ulap = self._eval_u_core(r_carts, zone)
    phase, k_cart, _ = self._phase_in_zone(r_carts, zone)
    k2 = jnp.sum(k_cart * k_cart, axis=1)[:, None]
    return (ulap - k2 * u) * phase


def _bspline_fused_without_cross_term(self, r_carts):
    """Mutant fused kernel: same dropped ``2 i k . grad u`` term."""
    zone = "ao_grad_lap"
    u, ux, uy, uz, ulap = self._eval_u_core(r_carts, zone)
    phase, k_cart, imag_unit = self._phase_in_zone(r_carts, zone)
    val = (u * phase).astype(get_complex_dtype_jnp("ao_eval"))
    gx = (ux + imag_unit * k_cart[:, 0, None] * u) * phase
    gy = (uy + imag_unit * k_cart[:, 1, None] * u) * phase
    gz = (uz + imag_unit * k_cart[:, 2, None] * u) * phase
    k2 = jnp.sum(k_cart * k_cart, axis=1)[:, None]
    lap = (ulap - k2 * u) * phase
    return val, gx, gy, gz, lap


_ORIGINAL_PGTO_LAPLACIAN = _ComplexBlochPGTOMixin._compute_AOs_laplacian
_ORIGINAL_PGTO_FUSED = _ComplexBlochPGTOMixin._compute_AOs_value_grad_lap


def _pgto_laplacian_double_phase(self, r_carts):
    """Mutant: the wrap-restore phase applied twice to the Bloch pGTO Laplacian."""
    return _ORIGINAL_PGTO_LAPLACIAN(self, r_carts) * self._restore_phase(r_carts, "ao_grad_lap")[None, :]


def _pgto_fused_double_phase(self, r_carts):
    """Mutant fused kernel: the wrap-restore phase applied twice to the Laplacian."""
    val, gx, gy, gz, lap = _ORIGINAL_PGTO_FUSED(self, r_carts)
    return val, gx, gy, gz, lap * self._restore_phase(r_carts, "ao_grad_lap")[None, :]


def _install_bspline_cross_term_mutant(monkeypatch) -> None:
    monkeypatch.setattr(AOs_bspline_complex_data, "_compute_AOs_laplacian", _bspline_laplacian_without_cross_term)
    monkeypatch.setattr(AOs_bspline_complex_data, "_compute_AOs_value_grad_lap", _bspline_fused_without_cross_term)


def _install_pgto_double_phase_mutant(monkeypatch) -> None:
    monkeypatch.setattr(_ComplexBlochPGTOMixin, "_compute_AOs_laplacian", _pgto_laplacian_double_phase)
    monkeypatch.setattr(_ComplexBlochPGTOMixin, "_compute_AOs_value_grad_lap", _pgto_fused_double_phase)


@pytest.mark.light
@pytest.mark.parametrize("case", ["bspline_complex", "composite_complex"])
def test_negative_control_dropped_bloch_cross_term_fails_ao_laplacian(case: str, monkeypatch):
    """Dropping ``2 i k . grad u`` from the complex B-spline Laplacian turns the AD row red.

    The mutated leaf keeps its value and gradient, so the gradient comparisons
    still pass and the failure must come from the Laplacian row.
    """
    full_reference = None
    if is_mixed_precision_enabled():
        _, full_reference = mixed_vs_full(lambda: _ao_fidelity_case(case), _analytical_ao_outputs)
    aos_data, cell = _build_ao_case(case)
    r_carts = _straddling_electrons(cell, seed=20260927)
    clean = jax.tree_util.tree_map(np.asarray, _analytical_ao_outputs((aos_data, r_carts)))
    jax.clear_caches()
    _install_bspline_cross_term_mutant(monkeypatch)
    mutated = jax.tree_util.tree_map(np.asarray, _analytical_ao_outputs((aos_data, r_carts)))
    # A numerical negative control must retain all unaffected outputs, including
    # the fused gradients that the Laplacian assertion below would short-circuit.
    for actual, expected in zip(
        (mutated["value"], *mutated["grads"], *mutated["fused"][:4]),
        (clean["value"], *clean["grads"], *clean["fused"][:4]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    comparison = "mixed vs full" if is_mixed_precision_enabled() else "analytic vs AD"
    with pytest.raises(AssertionError, match=rf"{case} lap {comparison}"):
        _compare_ao_derivatives(case, aos_data, r_carts, full_reference=full_reference)
    monkeypatch.undo()
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("case", ["pgto_cart_l4_complex", "pgto_sphe_l4_complex"])
def test_negative_control_double_restore_phase_fails_ao_laplacian(case: str, monkeypatch):
    """Applying the pGTO wrap-restore phase twice to the Laplacian turns the AD row red."""
    full_reference = None
    if is_mixed_precision_enabled():
        _, full_reference = mixed_vs_full(lambda: _ao_fidelity_case(case), _analytical_ao_outputs)
    aos_data, cell = _build_ao_case(case)
    r_carts = _straddling_electrons(cell, seed=20260927)
    jax.clear_caches()
    _install_pgto_double_phase_mutant(monkeypatch)
    comparison = "mixed vs full" if is_mixed_precision_enabled() else "analytic vs AD"
    with pytest.raises(AssertionError, match=rf"{case} lap {comparison}"):
        _compare_ao_derivatives(case, aos_data, r_carts, full_reference=full_reference)
    monkeypatch.undo()
    jax.clear_caches()


# ---------------------------------------------------------------------------
# Complex twist kinetic energy vs the Hessian of the complex Psi.
# ---------------------------------------------------------------------------

#: case id -> (committed fixture stem, attach J1 + J2). No J3 at twist.
_KINETIC_CASES = {
    "lithium_k025-jnone": ("lithium_k025.h5", False),
    "lithium_k025-j12": ("lithium_k025.h5", True),
    "H2_pbc_ae_k025_cart-jnone": ("H2_pbc_ae_k025_cart.h5", False),
    "H2_pbc_ae_k025_cart-j12": ("H2_pbc_ae_k025_cart.h5", True),
}


@jax.jit
def _kinetic_from_complex_psi_hessian(wavefunction_data, r_up, r_dn):
    """Per-electron ``-1/2 (lap_i Psi) / Psi`` from the AD Hessian of the complex ``Psi``."""
    n_up = r_up.shape[0]
    positions = jnp.concatenate((r_up, r_dn))

    def psi(points):
        return evaluate_wavefunction(wavefunction_data, points[:n_up], points[n_up:])

    hessian = jax.jacfwd(jax.jacfwd(psi))(positions)  # (N, 3, N, 3), complex
    return -0.5 * jnp.einsum("ijij->i", hessian) / psi(positions)


def _kinetic_electrons(ham, seed: int):
    """Electrons near the ions; electron 0 is wrapped into the home cell, every other one shifted by a lattice vector."""
    geminal = ham.wavefunction_data.geminal_data
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    n_up, n_dn = geminal.num_electron_up, geminal.num_electron_dn
    rng = np.random.default_rng(seed)
    centres = positions[np.arange(n_up + n_dn) % positions.shape[0]]
    r = centres + rng.normal(scale=0.7, size=(n_up + n_dn, 3))
    r[0] -= np.floor(r[0] @ np.linalg.inv(cell)) @ cell
    shifts = 2.0 * np.array([cell[0], -cell[1], cell[2], -cell[0]])
    for j in range(1, n_up + n_dn, 2):
        r[j] += shifts[(j // 2) % len(shifts)]
    return r[:n_up], r[n_up:], cell


def _kinetic_fidelity_case(case):
    stem, with_j = _KINETIC_CASES[case]
    ham = _build_ham_for_cell(stem, with_j, None, "SD")
    r_up, r_dn, _ = _kinetic_electrons(ham, seed=20260927)
    return ham.wavefunction_data, r_up, r_dn


def _analytical_kinetic_outputs(case):
    wavefunction, r_up, r_dn = case
    psi = np.asarray(evaluate_wavefunction(wavefunction, r_up, r_dn))
    assert np.all(np.isfinite(psi)) and np.all(np.abs(psi) > 0), "kinetic fixture requires finite nonzero Psi"
    per_electron = np.concatenate([np.asarray(t) for t in compute_kinetic_energy_all_elements(wavefunction, r_up, r_dn)])
    assert per_electron.shape == (r_up.shape[0] + r_dn.shape[0],)
    # Keep the old scalar conversion: array broadcasting must not certify a total.
    total = complex(np.asarray(compute_kinetic_energy(wavefunction, r_up, r_dn)))
    return {"per_electron": per_electron, "total": np.asarray(total)}


def _compare_kinetic(case_id: str, wavefunction_data, r_up, r_dn, *, full_reference=None, mixed_result=None) -> float:
    """Full Hessian oracle or mixed normalized-Laplacian fidelity, with same-mode total/sum closure."""
    if is_mixed_precision_enabled():
        if full_reference is None:
            mixed, full = mixed_vs_full(lambda: _kinetic_fidelity_case(case_id), _analytical_kinetic_outputs)
        else:
            mixed = _analytical_kinetic_outputs((wavefunction_data, r_up, r_dn)) if mixed_result is None else mixed_result
            full = full_reference
        worst = assert_mixed_orbital_derivative_fidelity(
            mixed["per_electron"], full["per_electron"], f"{case_id} per-electron kinetic mixed vs full"
        )
        # The scalar/sum identity remains within mixed precision with its old budget.
        atol, rtol = get_tolerance_min(_KINETIC_ZONES, "strict")
        np.testing.assert_allclose(
            mixed["total"],
            np.sum(mixed["per_electron"]),
            atol=atol,
            rtol=rtol,
            err_msg=f"{case_id} same-mode total kinetic sum",
        )
        return worst
    atol, rtol = get_tolerance_min(_KINETIC_ZONES, "strict")
    actual = np.concatenate([np.asarray(t) for t in compute_kinetic_energy_all_elements(wavefunction_data, r_up, r_dn)])
    reference = np.asarray(_kinetic_from_complex_psi_hessian(wavefunction_data, r_up, r_dn))
    assert actual.shape == reference.shape == (r_up.shape[0] + r_dn.shape[0],)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), f"{case_id}: non-finite kinetic"
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=f"{case_id} per-electron kinetic vs AD")
    total = complex(np.asarray(compute_kinetic_energy(wavefunction_data, r_up, r_dn)))
    np.testing.assert_allclose(
        total, complex(np.sum(reference)), atol=atol, rtol=rtol, err_msg=f"{case_id} total kinetic vs AD"
    )
    return float(np.max(np.abs(actual - reference)))


@pytest.mark.light
@pytest.mark.parametrize("case", tuple(_KINETIC_CASES))
def test_complex_twist_kinetic_matches_complex_psi_hessian(case):
    """Analytic complex kinetic (all-elements and scalar) equals ``-1/2 lap Psi / Psi`` by AD of ``Psi``.

    Committed twist fixtures, Bloch pGTO with complex MO coefficients, with and
    without J1 + J2; electrons leave the home cell. Re and Im are compared, and
    Im must exceed the element-wise allclose envelope at least tenfold somewhere.
    """
    stem, with_j = _KINETIC_CASES[case]
    case_id = case
    ham = _build_ham_for_cell(stem, with_j, None, "SD")
    wf = ham.wavefunction_data
    assert is_complex_aos(wf.geminal_data.orb_data_up_spin.aos_data), f"{case_id}: fixture is not a complex twist leaf"
    if with_j:
        jd = wf.jastrow_data
        assert jd.jastrow_one_body_data is not None and jd.jastrow_two_body_data is not None
        assert jd.jastrow_three_body_data is None
    r_up, r_dn, cell = _kinetic_electrons(ham, seed=20260927)
    _assert_straddles(np.concatenate((r_up, r_dn)), cell)
    atol, rtol = get_tolerance_min(_KINETIC_ZONES, "strict")
    full_reference = mixed_result = None
    if is_mixed_precision_enabled():
        mixed_result, full_reference = mixed_vs_full(lambda: _kinetic_fidelity_case(case_id), _analytical_kinetic_outputs)
        reference = np.asarray(full_reference["per_electron"])
    else:
        reference = np.asarray(_kinetic_from_complex_psi_hessian(wf, r_up, r_dn))
    # Non-vacuity: dropping Im must miss the unchanged liveness envelope by a wide margin.
    envelope = atol + rtol * np.abs(reference)
    assert np.max(np.abs(reference.imag) / envelope) > _KINETIC_IM_MARGIN, f"{case_id}: Im kinetic vacuous"
    deviation = _compare_kinetic(case_id, wf, r_up, r_dn, full_reference=full_reference, mixed_result=mixed_result)
    print(f"{case_id}: max derivative deviation = {deviation:.3e}, max |Im T_ref| = {np.max(np.abs(reference.imag)):.3e}")
    jax.clear_caches()


@pytest.mark.light
def test_negative_control_double_restore_phase_fails_twist_kinetic(monkeypatch):
    """The pGTO double-restore-phase mutant makes the twist kinetic comparison fail."""
    full_reference = None
    if is_mixed_precision_enabled():
        _, full_reference = mixed_vs_full(lambda: _kinetic_fidelity_case("lithium_k025-jnone"), _analytical_kinetic_outputs)
    ham = _build_ham_for_cell("lithium_k025.h5", False, None, "SD")
    wf = ham.wavefunction_data
    r_up, r_dn, _ = _kinetic_electrons(ham, seed=20260927)
    jax.clear_caches()
    _install_pgto_double_phase_mutant(monkeypatch)
    comparison = "mixed vs full" if is_mixed_precision_enabled() else "vs AD"
    with pytest.raises(AssertionError, match=f"lithium_k025-jnone per-electron kinetic {comparison}"):
        _compare_kinetic("lithium_k025-jnone", wf, r_up, r_dn, full_reference=full_reference)
    monkeypatch.undo()
    jax.clear_caches()

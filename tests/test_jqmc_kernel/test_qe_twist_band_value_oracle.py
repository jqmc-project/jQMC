"""QE -> twist-average route: converted MO values vs independent QE band values at k != Gamma.

The per-twist decomposition oracle in
``test_qe_twist_decomposition_nonpgto_mpi`` compares the twist-average driver
with standalone chains, but both sides consume the SAME converted AO leaf.  A
wrongly placed ``jnp.real`` on the complex AO leaf (plane wave, B-spline, or
the Bloch pGTO child of a composite) at a non-Gamma twist would therefore
agree on both sides and pass.  This module compares the leaf output of the
twist route with an independent reference built only from the QE ``*.save``.

Route under test: the committed ``Li_k025`` save (bcc Li, 2 atoms, one
occupied band per spin, fractional k = (1/4, 1/4, 1/4)) is converted with
``convert_qe_to_twist_average_input`` (default per-kind options) and loaded
with ``load_twist``.  The converted up-spin MOs are evaluated with production
``compute_MOs``.

Independent reference: ``psi_b(r) = sum_G c_{b,G} exp(i (k + G) . r)`` in
numpy, with ``c_{b,G}``, the Miller indices and the Cartesian ``k`` read from
``wfc1.hdf5`` by ``load_qe_reference``.  This is the same direct band sum the
existing QE fidelity tests use (``_pw_band_values_in_zones`` in
``test_qe_to_jqmc.py`` and the Li_k025 dn-channel test), extended by the
twist ``k``.  Precision: the QE data are fp64 inputs and the band sum is evaluated
in the production precision zones -- the phase argument ``(k + G) . r`` is
reconstructed in fp64 and cast to ``ao_eval``, the plane-wave values are
evaluated in ``ao_eval`` (complex64 in mixed) and the coefficient
contraction in ``mo_eval`` -- so mixed is judged within its own precision;
in full every step is fp64.

Scale convention: VMC is invariant under a per-band uniform rescale, so each
band is compared after the least-squares complex scalar ``c_b`` (rescale plus
global phase) is removed, as the lossless-kind comparison does with a real
scalar at Gamma.  A zero or nonfinite band is rejected before fitting.  A
complex scalar cannot absorb a real projection of the leaf: a k != Gamma
Bloch band is not a global phase times a real function.

What is asserted, per kind:

* ``pw``, ``bspline``, ``pgto+bspline`` (lossless kinds): the converted MO
  values at 16 fixed off-lattice points reproduce the QE band values, per
  band after the complex rescale.  ``pw`` is exact up to fp round-off
  (``RTOL_PW_PROJECTION_FP64`` in full, the strict AO/MO eval tolerance in
  mixed); the B-spline kinds carry the spline interpolation error at the
  twist route's default ``mesh_factor=0.75`` (measured 3.4e-7 in full mode;
  1.4e-5 at mesh_factor=0.5) and use the
  composite per-band gate ``RTOL_QE_COMPOSITE_PER_BAND_REGRESSION_GATE``.
* ``pgto+pw`` (default kind, not lossless): only the
  PW-represented component is asserted.  The converted composite MO is
  sampled by ``compute_MOs`` on a uniform grid of twice the QE FFT grid,
  the Bloch phase ``exp(-i k.r)`` is removed and the result is Fourier
  transformed.  The coefficients on the G set of the PW child
  (``|k + G| <= g_cut`` with the route's default
  ``g_cut = 0.5 * sqrt(2 * ecutwfc)``; the set is checked to be exactly the
  converted child's Miller set) must equal QE's ``c_G`` after the complex
  rescale, within ``RTOL_QE_PW_RECONSTRUCTION_LOOSE``.  The mixed
  projection fits that set exactly; what remains is grid aliasing of the
  pGTO high-G tail (measured 1.4e-7 at the 2x grid in full mode).
  The stored ``c_G`` are not compared directly with the analysed production
  values.  The reference coefficients are the same round trip applied to
  the QE band: ``sum_G c_G exp(i (k + G) . r)`` over that G set is
  evaluated on the same grid in the production zones (as for the other
  kinds) and passed through the same fp64 analysis transform.  The
  analysis transform itself consumes ``mo_eval`` output and stays fp64 on
  both sides; in full the round trip returns ``c_G`` to fp64 round-off.
  NOT asserted for ``pgto+pw``: coefficients with ``|k + G| > g_cut``
  (fitted by the pGTOs only; measured 6.3e-4 over the full wfc sphere),
  content beyond the wfc sphere (the pGTO high-G leak the finite PW set
  cannot cancel), and any pointwise real-space agreement with QE.

Non-vacuity, per kind: the aligned converted values carry an imaginary part
of at least ``_IM_FRACTION_FLOOR`` of their peak modulus, and the same oracle
applied to their real part (a real-projected orbital) must exceed the
tolerance.  The negative control patches ``jnp.real`` onto the complex leaf
evaluation and requires the oracle's own residual assertion to fail.
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

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    RTOL_PW_PROJECTION_FP64,
    RTOL_QE_COMPOSITE_PER_BAND_REGRESSION_GATE,
    RTOL_QE_PW_RECONSTRUCTION_LOOSE,
)
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_cart_pgto_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    is_complex_aos,
)
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_qe_to_jqmc import _pw_band_values_in_zones

_SAVE = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"
_UPF = _SAVE / "Li.ccECP.upf"
_KINDS = ("pw", "bspline", "pgto+pw", "pgto+bspline")
_N_PROBE = 16
_PROBE_SEED = 2026
# Floor on max|Im(aligned psi)| / max|psi| of the converted band.  The Li_k025
# band measures 0.94; a real (or Gamma-like) orbital gives ~0.
_IM_FRACTION_FLOOR = 0.1
# The pgto+pw sampling grid is this multiple of the QE FFT grid per axis.
_GRID_MULTIPLE = 2
# Grid points per compute_MOs call (bounds the AO block memory; the grid is
# 72^3 points for Li_k025).
_GRID_CHUNK = 20000


def _tolerance(kind: str) -> float:
    if kind == "pw":
        if is_mixed_precision_enabled():
            return float(get_tolerance_min(("ao_eval", "mo_eval"), "strict")[1])
        return RTOL_PW_PROJECTION_FP64
    if kind == "pgto+pw":
        return RTOL_QE_PW_RECONSTRUCTION_LOOSE
    return RTOL_QE_COMPOSITE_PER_BAND_REGRESSION_GATE


def _complex_rescale_residual(ref: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Per band (rows): relative residual after removing the least-squares complex scalar.

    Both arrays must be finite, nonempty, two-dimensional and of equal shape,
    with a nonzero norm in every band; a zero band is not an admissible scale.
    Columns are normalized by their peak modulus before fitting.
    """
    ref, vals = np.asarray(ref), np.asarray(vals)
    assert ref.ndim == vals.ndim == 2, "band value oracle: expected two-dimensional arrays"
    assert ref.shape == vals.shape, f"band value oracle: shapes differ {ref.shape} vs {vals.shape}"
    assert ref.size > 0, "band value oracle: sampled arrays must be nonempty"
    assert np.all(np.isfinite(ref)), "band value oracle: nonfinite reference values"
    assert np.all(np.isfinite(vals)), "band value oracle: nonfinite converted values"
    out = np.empty(ref.shape[0], dtype=np.float64)
    for b in range(ref.shape[0]):
        assert np.any(ref[b]), "band value oracle: reference band has zero norm"
        assert np.any(vals[b]), "band value oracle: converted band has zero norm"
        x = ref[b] / np.max(np.abs(ref[b]))
        y = vals[b] / np.max(np.abs(vals[b]))
        c = np.vdot(x, y) / np.vdot(x, x)
        out[b] = float(np.linalg.norm(y - c * x) / np.linalg.norm(y))
    assert np.all(np.isfinite(out)), "band value oracle: nonfinite residual"
    return out


def _phase_aligned(vals: np.ndarray) -> np.ndarray:
    """Rotate each band (row) so its largest-modulus sample is real positive."""
    out = np.array(vals, dtype=np.complex128)
    for b in range(out.shape[0]):
        j = int(np.argmax(np.abs(out[b])))
        out[b] = out[b] * (np.abs(out[b, j]) / out[b, j])
    return out


def _probe_points(ref) -> np.ndarray:
    rng = np.random.default_rng(_PROBE_SEED)
    return rng.uniform(0.0, 1.0, size=(_N_PROBE, 3)) @ np.asarray(ref.cell.a, dtype=np.float64)


def _qe_band_values(ref, band_idx, r_carts, g_mask=None) -> np.ndarray:
    """Independent QE band sum ``sum_G c_G exp(i (k + G) . r)`` in the production zones; ``(n_band, n_probe)``.

    ``g_mask`` restricts the sum to a subset of the wfc G set. See
    ``_pw_band_values_in_zones`` (plane-wave values in ``ao_eval``,
    contraction in ``mo_eval``).
    """
    kg = ref.wfc.millers.astype(np.float64) @ np.asarray(ref.cell.b) + np.asarray(ref.wfc.k_point)
    coeffs = ref.wfc.coeffs[np.asarray(band_idx)]
    if g_mask is not None:
        kg, coeffs = kg[g_mask], coeffs[:, g_mask]
    return _pw_band_values_in_zones(coeffs, kg, r_carts)


def _pw_subset(ref) -> np.ndarray:
    """Boolean mask over the wfc G set: ``|k + G| <= 0.5 * sqrt(2 * ecutwfc)`` (route default)."""
    g_cut = 0.5 * float(np.sqrt(2.0 * ref.ecutwfc))
    kg = ref.wfc.millers.astype(np.float64) @ np.asarray(ref.cell.b) + np.asarray(ref.wfc.k_point)
    return np.linalg.norm(kg, axis=1) <= g_cut


def _grid_fourier_coefficients(ref, vals_on_grid: np.ndarray, n_grid: tuple[int, int, int]) -> np.ndarray:
    """Fourier coefficients of ``exp(-i k.r) psi(r)`` at the wfc Miller set; ``(n_band, n_g)``."""
    r = _grid_points(ref, n_grid)
    u = vals_on_grid * np.exp(-1j * (r @ np.asarray(ref.wfc.k_point)))[None, :]
    m = ref.wfc.millers
    out = np.empty((u.shape[0], m.shape[0]), dtype=np.complex128)
    for b in range(u.shape[0]):
        spec = np.fft.fftn(u[b].reshape(n_grid)) / float(np.prod(n_grid))
        out[b] = spec[m[:, 0] % n_grid[0], m[:, 1] % n_grid[1], m[:, 2] % n_grid[2]]
    return out


def _grid_points(ref, n_grid: tuple[int, int, int]) -> np.ndarray:
    frac = np.stack(np.meshgrid(*(np.arange(n) / n for n in n_grid), indexing="ij"), axis=-1).reshape(-1, 3)
    return frac @ np.asarray(ref.cell.a, dtype=np.float64)


def _occupied_band_indices(ref, num_mo: int) -> np.ndarray:
    band_idx = np.asarray(ref.bands.occupied_band_indices())
    assert band_idx.size == num_mo, f"band value oracle: {band_idx.size} occupied QE bands vs {num_mo} converted MOs"
    return band_idx


def _assert_band_value_oracle(kind: str, ham, ref) -> dict:
    """Run the value oracle for one kind; returns the measured numbers."""
    from jqmc.molecular_orbital import compute_MOs

    mos = ham.wavefunction_data.geminal_data.orb_data_up_spin
    band_idx = _occupied_band_indices(ref, int(mos.num_mo))
    tol = _tolerance(kind)

    if kind == "pgto+pw":
        n_grid = tuple(_GRID_MULTIPLE * int(n) for n in ref.fft_grid)
        r = _grid_points(ref, n_grid)
        vals = np.concatenate(
            [np.asarray(compute_MOs(mos, r[i : i + _GRID_CHUNK])) for i in range(0, r.shape[0], _GRID_CHUNK)], axis=1
        )
        assert np.all(np.isfinite(vals)), "band value oracle: nonfinite converted grid values"
        mask = _pw_subset(ref)
        got = _grid_fourier_coefficients(ref, vals, n_grid)[:, mask]
        # The reference coefficients take the same value -> analysis round
        # trip as the production values (zone-dtype band sum, fp64 transform).
        ref_vals = _qe_band_values(ref, band_idx, r, g_mask=mask)
        want = _grid_fourier_coefficients(ref, ref_vals, n_grid)[:, mask]
        resid = _complex_rescale_residual(want, got)
        vals_r = _phase_aligned(vals).real.astype(np.complex128)
        resid_real = _complex_rescale_residual(want, _grid_fourier_coefficients(ref, vals_r, n_grid)[:, mask])
    else:
        r = _probe_points(ref)
        vals = np.asarray(compute_MOs(mos, r))
        want = _qe_band_values(ref, band_idx, r)
        resid = _complex_rescale_residual(want, vals)
        resid_real = _complex_rescale_residual(want, _phase_aligned(vals).real.astype(np.complex128))

    assert np.max(resid) < tol, (
        f"band value oracle: kind={kind!r} converted MO differs from the QE band per band after "
        f"complex rescale: rel_resid={resid} exceeds tol={tol:.1e}"
    )

    aligned = _phase_aligned(vals)
    im_fraction = np.max(np.abs(aligned.imag), axis=1) / np.max(np.abs(aligned), axis=1)
    assert np.all(np.isfinite(im_fraction)), "band value oracle: nonfinite imaginary fraction"
    assert np.all(im_fraction > _IM_FRACTION_FLOOR), (
        f"kind={kind!r}: converted band imaginary part is vacuous: im_fraction={im_fraction}"
    )
    assert np.all(resid_real > tol), (
        f"kind={kind!r}: the oracle does not distinguish a real-projected band: rel_resid(real)={resid_real}, tol={tol:.1e}"
    )
    return {"resid": resid, "resid_real": resid_real, "im_fraction": im_fraction, "tol": tol}


@pytest.fixture(scope="module")
def qe_reference():
    from jqmc.qe_wrapper import load_qe_reference

    return load_qe_reference(str(_SAVE))


@pytest.fixture(scope="module")
def twist_route_files(tmp_path_factory):
    """Convert Li_k025 once per kind through the twist-average route; kind -> HDF5 path."""
    from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
    from jqmc.upf_wrapper import parse_gaussinfo

    ecp = parse_gaussinfo(str(_UPF))
    out_dir = tmp_path_factory.mktemp("qe_twist_band_oracle")
    paths = {}
    for kind in _KINDS:
        path = str(out_dir / f"li_k025_{kind.replace('+', '_')}.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_SAVE),
            output_path=path,
            upf_paths={"Li": str(_UPF)},
            kind=kind,
            ecp_by_element={"Li": ecp},
            j1_param=None,
            j2_param=None,
        )
        paths[kind] = path
    return paths


def _load(twist_route_files, kind, ref):
    from jqmc._twist_average import load_twist

    ham = load_twist(twist_route_files[kind], 0)
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert is_complex_aos(aos), f"kind={kind}: the k025 AO leaf must be complex, got {type(aos).__name__}"
    assert np.any(np.asarray(ref.wfc.k_point)), "Li_k025 must be a non-Gamma twist"
    if kind == "pgto+pw":
        assert isinstance(aos, AOs_composite_data)
        pw_child = aos.children[1]
        assert isinstance(pw_child, AOs_plane_wave_complex_data)
        child_millers = {tuple(int(v) for v in m) for m in np.asarray(pw_child._g_indices).reshape(-1, 3)}
        subset = {tuple(int(v) for v in m) for m in ref.wfc.millers[_pw_subset(ref)]}
        assert child_millers == subset, (
            f"PW-represented set mismatch: child has {len(child_millers)} G, |k+G|<=g_cut subset has {len(subset)}"
        )
    return ham


@pytest.mark.heavy
@pytest.mark.parametrize("kind", _KINDS)
def test_qe_twist_route_mo_values_match_qe_bands_at_non_gamma(twist_route_files, qe_reference, kind):
    """Twist-route converted MOs equal the independent QE band values at Li_k025 (see module docstring)."""
    import jax

    jax.clear_caches()
    ham = _load(twist_route_files, kind, qe_reference)
    _assert_band_value_oracle(kind, ham, qe_reference)


# Negative-control case -> (kind, complex leaf class whose evaluation is real-projected).
_CONTROL_CASES = {
    "pw-pw": ("pw", AOs_plane_wave_complex_data),
    "bspline-bspline": ("bspline", AOs_bspline_complex_data),
    "pgto_pw-pw": ("pgto+pw", AOs_plane_wave_complex_data),
    "pgto_pw-pgto": ("pgto+pw", AOs_cart_pgto_complex_data),
    "pgto_bspline-bspline": ("pgto+bspline", AOs_bspline_complex_data),
    "pgto_bspline-pgto": ("pgto+bspline", AOs_cart_pgto_complex_data),
}


@pytest.mark.heavy
@pytest.mark.parametrize("case", sorted(_CONTROL_CASES))
def test_qe_twist_band_value_oracle_rejects_real_projected_leaf(twist_route_files, qe_reference, case, monkeypatch):
    """Negative control: ``jnp.real`` on the complex leaf evaluation fails the oracle's residual assertion."""
    import jax
    import jax.numpy as jnp

    kind, leaf_cls = _CONTROL_CASES[case]
    ham = _load(twist_route_files, kind, qe_reference)
    original = leaf_cls._compute_AOs
    calls = []

    def _real_projected(self, r_carts):
        calls.append(type(self).__name__)
        return jnp.real(original(self, r_carts))

    monkeypatch.setattr(leaf_cls, "_compute_AOs", _real_projected)
    jax.clear_caches()
    try:
        with pytest.raises(AssertionError, match=r"band value oracle: kind=.* exceeds tol="):
            _assert_band_value_oracle(kind, ham, qe_reference)
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    assert calls, f"kind={kind}: the patched {leaf_cls.__name__}._compute_AOs was never reached"

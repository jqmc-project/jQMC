"""Unit-cell MO normalization and pGTO band-order orthonormalization.

Every QE conversion kind emits MOs normalized over the unit cell (the
``Omega^{-1/2}`` of ``psi_n = Omega^{-1/2} sum_G c_{n,G} e^{i(G+k).r}`` is carried
by the coefficients). The pure ``pgto`` kind additionally orthonormalizes each
``(k, sigma)`` block in the periodic AO metric ``S = compute_overlap_matrix``
of the built leaf by band-order Cholesky, ``C' = conj(L)^{-1} C`` with
``conj(C) S C^T = L L^H``. The conversion is an fp64 preprocessing step in both
precision modes; each mode is judged on its own (no mixed-vs-full verdict).
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

from contextlib import contextmanager
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from scipy.special import gamma

from jqmc import qe_to_jqmc
from jqmc._precision import get_dtype_np, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PW_PROJECTION_FP64,
    MO_ORTHONORMALITY_WARNING_THRESHOLD,
    RTOL_PW_PROJECTION_FP64,
)
from jqmc.atomic_orbital import AOs_composite_data, AOs_plane_wave_data, compute_overlap_matrix
from jqmc.determinant import compute_det_geminal_all_elements
from jqmc.pw_pgto_projection import (
    CartPrimitive,
    ProjectionResult,
    build_gto_pw_matrix,
    project_pw_to_mixed,
    project_pw_to_pgto,
    project_pw_to_pgto_spline,
    project_pw_to_pw,
    project_pw_to_spline,
)
from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction, load_qe_reference
from jqmc.upf_wrapper import UPFExponentInit
from tests._test_data_cache import qe_example_files_root

pytestmark = [pytest.mark.without_chdir]

_QE = qe_example_files_root()
_SAVES = {
    "li-gamma": (_QE / "Li_gamma" / "out" / "li.save", "Li.ccECP.upf", False),
    "li-k025": (_QE / "Li_k025" / "out" / "li_k025.save", "Li.ccECP.upf", False),
    "o-lsda-gamma": (_QE / "O-lsda_gamma" / "out" / "o.save", "O.ccECP.upf", True),
    "o-lsda-k025": (_QE / "O-lsda_k025" / "out" / "o_k025.save", "O.ccECP.upf", True),
}


# ---------------------------------------------------------------------------
# Shared fixtures (process-local caches; the projections are pure functions)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _reference(fixture, spin=None):
    save, _upf, _lsda = _SAVES[fixture]
    return load_qe_reference(str(save)) if spin is None else load_qe_reference(str(save), spin=spin)


@lru_cache(maxsize=None)
def _shells(fixture):
    save, upf, _lsda = _SAVES[fixture]
    ref = _reference(fixture, "up" if _SAVES[fixture][2] else None)
    element = ref.atom_elements[0].lower()
    strategy = UPFExponentInit({element: str(save / upf)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
    return tuple(tuple(strategy.shells_for_element(label)) for label in ref.atom_labels)


@lru_cache(maxsize=None)
def _projection(fixture, kind, spin=None, occ_only=False):
    ref = _reference(fixture, spin)
    shells = [list(s) for s in _shells(fixture)] if kind != "pw" and kind != "bspline" else None
    if kind == "pgto":
        return project_pw_to_pgto(ref, shells, occ_only=occ_only)
    if kind == "pw":
        return project_pw_to_pw(ref, g_cut=None, occ_only=occ_only)
    if kind == "bspline":
        return project_pw_to_spline(ref, mesh_factor=0.5, order=5, occ_only=occ_only)
    if kind == "pgto+pw":
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        return project_pw_to_mixed(ref, shells, g_cut=0.5 * g_max, drop_diffuse=True, occ_only=occ_only)
    if kind == "pgto+bspline":
        return project_pw_to_pgto_spline(ref, shells, mesh_factor=0.5, order=5, occ_only=occ_only)
    raise ValueError(kind)


@contextmanager
def _captured_mo_blocks():
    """Spy on the converter's geminal packaging to capture the FULL MO blocks.

    ``build_wavefunction_from_projection`` hands the complete band-ordered MO
    block (before the occupied rows are selected for the determinant) to
    ``_make_geminal_from_mos``; the spy records that ``MOs_data`` (and the
    LSDA dn block) and then calls the production function unchanged.
    """
    captured = {}
    original = qe_to_jqmc._make_geminal_from_mos

    def spy(mos_data, occupations, n_electrons_total, **kwargs):
        captured["up"] = mos_data
        captured["dn"] = kwargs.get("mos_data_dn")
        return original(mos_data, occupations, n_electrons_total, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qe_to_jqmc, "_make_geminal_from_mos", spy)
        yield captured


def _convert(fixture, kind, *, orthonormalize=True, occ_only=False):
    """``(geminal, full up MOs_data, full dn MOs_data | None)`` from the public builder."""
    lsda = _SAVES[fixture][2]
    ref = _reference(fixture, "up" if lsda else None)
    proj = _projection(fixture, kind, "up" if lsda else None, occ_only)
    kwargs = {}
    if lsda:
        kwargs = dict(
            qe_reference_dn=_reference(fixture, "dw"), projection_result_dn=_projection(fixture, kind, "dw", occ_only)
        )
    with _captured_mo_blocks() as captured:
        _, _, geminal = qe_to_jqmc.build_wavefunction_from_projection(
            ref, proj, kind=kind, orthonormalize=orthonormalize, **kwargs
        )
    return geminal, captured["up"], captured["dn"]


def _gram(mos, overlap=None):
    c = np.asarray(mos.mo_coefficients)
    s = np.asarray(compute_overlap_matrix(mos.aos_data)) if overlap is None else np.asarray(overlap)
    return c.conj() @ s @ c.T


def _assert_orthonormal(mos, overlap=None):
    gram = _gram(mos, overlap)
    np.testing.assert_allclose(
        gram, np.eye(gram.shape[0]), atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64, equal_nan=False
    )


def _cholesky_transform(old_coefficients, overlap):
    """Independent ``conj(L)^{-1}`` for the band-ordered block (NumPy only)."""
    gram = old_coefficients.conj() @ overlap @ old_coefficients.T
    lower = np.linalg.cholesky(0.5 * (gram + gram.conj().T))
    return np.linalg.inv(lower.conj())


# ---------------------------------------------------------------------------
# 1. The orthonormalization metric: S_AO of the built leaf vs the Fourier Gram
# ---------------------------------------------------------------------------
_TRICLINIC = np.array([[5.2, 0.0, 0.0], [0.9, 5.6, 0.0], [-0.5, 0.7, 6.1]])
_CENTERS = (np.array([0.4, 0.3, 0.9]), np.array([2.9, 3.4, 4.6]))
_PRIMITIVE_SPECS = ((0.45, (0, 0, 0)), (1.3, (0, 0, 0)), (0.8, (1, 0, 0)), (0.8, (0, 1, 0)), (0.8, (0, 0, 1)))
_PRIMITIVE_SPECS_D = ((0.9, (2, 0, 0)), (0.9, (1, 1, 0)), (0.9, (0, 1, 1)))


def _synthetic_leaf(k_frac):
    """Two-centre triclinic pGTO leaf built by the converter's own pGTO builder."""
    cell = QECell(_TRICLINIC)
    prims = []
    for center in _CENTERS:
        for exponent, ang in _PRIMITIVE_SPECS + _PRIMITIVE_SPECS_D:
            prims.append(CartPrimitive(center=tuple(center), ang=ang, exponent=exponent))
    k_cart = np.asarray(k_frac) @ cell.b
    ref = QEReference(
        cell=cell,
        atom_labels=("Be", "Be"),
        atom_positions=np.stack(_CENTERS),
        ecutwfc=5.0,
        fft_grid=(8, 8, 8),
        bands=QEBands(np.array([-0.4]), np.ones(1), 2.0, None),
        wfc=QEWavefunction(np.zeros((1, 3), dtype=np.int64), np.ones((1, 1), complex), not np.any(k_frac), k_cart),
        atom_elements=("Be", "Be"),
    )
    projection = ProjectionResult(
        np.zeros(1), np.array([0]), np.ones(1), np.ones((1, len(prims)), complex), tuple(prims), len(prims), 0, 0, 1.0, 1.0, 1.0
    )
    structure, k_point_frac = qe_to_jqmc._build_structure_data(ref)
    leaf, _ = qe_to_jqmc._build_pgto_leaf_and_mo(projection, structure, ref.atom_positions, k_point_frac)
    return ref, prims, k_cart, leaf


def _independent_normalization(prim):
    """``N`` with ``integral (N x^nx y^ny z^nz e^{-a r^2})^2 = 1`` from Gamma functions."""
    alpha = prim.exponent
    norm2 = np.prod([gamma(n + 0.5) / (2.0 * alpha) ** (n + 0.5) for n in prim.ang])
    return 1.0 / np.sqrt(norm2)


@pytest.mark.light
@pytest.mark.parametrize("twist", ["gamma", "general-k"])
def test_periodic_pgto_overlap_equals_converged_fourier_gram(twist):
    """``compute_overlap_matrix`` of the converted pGTO leaf equals ``N G^H G N`` (phase and normalization check).

    Two Be centres in a triclinic cell carry s (0.45, 1.3), p (0.8) and d
    (0.9; xx, xy, yz) unnormalized Cartesian primitives. ``G`` is the
    projection's own analytic design matrix (``build_gto_pw_matrix``, the
    Fourier coefficients on cell-normalized plane waves) on every reciprocal
    vector with ``|G + k| <= sqrt(160 alpha_max)`` (the Gaussian pair tail
    beyond is below exp(-80)); ``N`` is an independent Gamma-function
    normalization. At Gamma and at k_frac = (0.27, -0.1, 0.13) the overlap of
    the leaf the converter builds (normalization, AO order and Bloch phase as
    jQMC evaluates them) matches within the fp64 projection pair tolerance; at
    the twist the conjugated overlap is rejected by the same comparison, so the
    metric the orthonormalization uses has the projection's phase convention.
    """
    k_frac = np.zeros(3) if twist == "gamma" else np.array([0.27, -0.1, 0.13])
    ref, prims, k_cart, leaf = _synthetic_leaf(k_frac)
    q_max = np.sqrt(160.0 * max(p.exponent for p in prims))
    reach = [int(np.ceil(q_max * np.linalg.norm(a) / (2.0 * np.pi))) + 1 for a in ref.cell.a]
    millers = np.array(list(product(*(range(-n, n + 1) for n in reach))), dtype=np.int64)
    millers = millers[np.linalg.norm(millers @ ref.cell.b + k_cart, axis=1) <= q_max]
    design = build_gto_pw_matrix(ref, prims, k_cart=k_cart, millers=millers)
    normalization = np.array([_independent_normalization(p) for p in prims])
    expected = normalization[:, None] * (design.conj().T @ design) * normalization[None, :]
    actual = np.asarray(compute_overlap_matrix(leaf))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64, equal_nan=False)
    if twist == "general-k":
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                actual.conj(), expected, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64, equal_nan=False
            )


# ---------------------------------------------------------------------------
# 2. / 5. / 7. Orthonormality, Gamma realness, and the L^{-1} negative control
# ---------------------------------------------------------------------------
@pytest.mark.light
@pytest.mark.parametrize("fixture", ["li-gamma", "li-k025", "o-lsda-gamma", "o-lsda-k025"])
def test_converted_pgto_mos_are_orthonormal_in_the_periodic_ao_metric(fixture):
    """The pure-pGTO converter output satisfies ``conj(C) S C^T = I`` per (k, sigma) block.

    All bands of the committed QE fixtures (Li 8 bands at Gamma and k = 1/4;
    O LSDA 6 + 6 bands) are projected (``occ_only=False``) and converted by
    ``build_wavefunction_from_projection(kind="pgto")``; the complete
    band-ordered blocks are captured just before the determinant rows are
    selected. With ``S = compute_overlap_matrix`` of each block's own leaf,
    the Gram matrix is the identity within the fp64 projection pair tolerance,
    and so are the occupied SD rows of both spin channels (including the
    time-reversed dn channel on the -k leaf). At Gamma the MOs stay real
    (float64). Before orthonormalization the same blocks are measurably
    non-orthonormal (the check is not vacuous).
    """
    geminal, up, dn = _convert(fixture, "pgto")
    _assert_orthonormal(up)
    if dn is not None:
        _assert_orthonormal(dn)
    for channel in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        _assert_orthonormal(channel)
    if fixture.endswith("gamma"):
        for mos in (up, dn, geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            if mos is not None:
                assert np.asarray(mos.mo_coefficients).dtype == np.float64
    _, up_raw, _ = _convert(fixture, "pgto", orthonormalize=False)
    gram_raw = _gram(up_raw)
    assert np.max(np.abs(gram_raw - np.eye(gram_raw.shape[0]))) > MO_ORTHONORMALITY_WARNING_THRESHOLD


@pytest.mark.light
@pytest.mark.parametrize("fixture", ["li-gamma", "li-k025"])
def test_inverse_cholesky_without_conjugate_breaks_orthonormality_only_at_general_k(fixture):
    """Negative control: ``L^{-1} C`` instead of ``conj(L)^{-1} C`` fails at a general k.

    With the production Cholesky solve replaced by the unconjugated factor
    (the documented trap), the converted Li block at k = 1/4 is no longer
    orthonormal and the identity comparison of the orthonormality test fails;
    at Gamma (real ``L``) the two are identical and the comparison still passes,
    which is why a Gamma-only test cannot detect this mistake.
    """
    original = scipy.linalg.solve_triangular

    def unconjugated(a, b, **kwargs):
        return original(np.conj(a), b, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qe_to_jqmc.scipy.linalg, "solve_triangular", unconjugated)
        _, up, _ = _convert(fixture, "pgto")
    if fixture == "li-gamma":
        _assert_orthonormal(up)
    else:
        with pytest.raises(AssertionError):
            _assert_orthonormal(up)


# ---------------------------------------------------------------------------
# 3. Lower-N subspace invariance for every N
# ---------------------------------------------------------------------------
@pytest.mark.light
@pytest.mark.parametrize("fixture", ["li-k025", "o-lsda-gamma", "o-lsda-k025"])
def test_orthonormalization_preserves_every_lowest_band_subspace(fixture):
    """For every N the projector onto the lowest N MOs is unchanged.

    ``P_N = X^T (conj(X) S X^T)^{-1} conj(X)`` with ``X = C[:N]`` is compared
    between ``orthonormalize=False`` and the default for all N of the
    converted blocks (each spin channel), relative to ``max|P_N|``, within the
    fp64 projection pair tolerance. A Loewdin (``M^{-1/2}``) transform of the
    same block, which mixes all bands, changes ``P_N`` for some N < n_band
    (negative control).
    """
    _, up_new, dn_new = _convert(fixture, "pgto")
    _, up_old, dn_old = _convert(fixture, "pgto", orthonormalize=False)
    for new, old in ((up_new, up_old), (dn_new, dn_old)):
        if new is None:
            continue
        s = np.asarray(compute_overlap_matrix(old.aos_data))
        c_old = np.asarray(old.mo_coefficients)
        c_new = np.asarray(new.mo_coefficients)
        gram = c_old.conj() @ s @ c_old.T
        w, u = np.linalg.eigh(0.5 * (gram + gram.conj().T))
        c_loewdin = (u @ np.diag(w**-0.5) @ u.conj().T).conj() @ c_old
        loewdin_changed = False
        for n in range(1, c_old.shape[0] + 1):

            def projector(x):
                return x.T @ np.linalg.solve(x.conj() @ s @ x.T, x.conj())

            p_old = projector(c_old[:n])
            scale = float(np.max(np.abs(p_old)))
            np.testing.assert_allclose(
                projector(c_new[:n]),
                p_old,
                atol=ATOL_PW_PROJECTION_FP64 * scale,
                rtol=RTOL_PW_PROJECTION_FP64,
                equal_nan=False,
                err_msg=f"N={n}",
            )
            loewdin_changed |= not np.allclose(
                projector(c_loewdin[:n]), p_old, atol=ATOL_PW_PROJECTION_FP64 * scale, rtol=RTOL_PW_PROJECTION_FP64
            )
        assert loewdin_changed


# ---------------------------------------------------------------------------
# 4. / 6. Constant log|Psi| and phase shift; LSDA shares S per k
# ---------------------------------------------------------------------------
def _configurations(geminal, positions, seed, count):
    """Deterministic ion-centred electron configurations (PCG64 ``seed``)."""
    rng = np.random.default_rng(seed)
    nu, nd = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    for _ in range(count):
        r_up = positions[np.arange(nu) % len(positions)] + rng.normal(scale=1.0, size=(nu, 3))
        r_dn = positions[np.arange(nd) % len(positions)] + rng.normal(scale=1.0, size=(nd, 3))
        yield r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("fixture", ["o-lsda-gamma", "o-lsda-k025"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_orthonormalization_changes_the_determinant_by_a_constant(fixture):
    """``Psi_orth / Psi_raw`` has the same modulus and phase at every configuration.

    The O LSDA triplet (4 up + 2 dn) is converted with and without
    orthonormalization from the occupied-band projections (the single-k CLI
    policy). At 24 PCG64(20261002) ion-centred configurations the determinant
    ratio is compared with ``det(conj(L_up)^{-1}) det(conj(L_dn)^{-1})``, where
    ``L`` is an independent NumPy Cholesky factor of each raw block's Gram
    matrix: ``log|ratio|`` equals ``log|prediction|`` and the complex
    ``ratio / prediction`` equals one (modulus and phase) within the
    active-mode strict ao/mo/det tolerance. The prediction is not 1 (the check is not vacuous).
    """
    geminal_new, up_new, dn_new = _convert(fixture, "pgto", occ_only=True)
    geminal_old, up_old, dn_old = _convert(fixture, "pgto", orthonormalize=False, occ_only=True)
    predicted = 1.0 + 0.0j
    for old in (up_old, dn_old):
        s = np.asarray(compute_overlap_matrix(old.aos_data))
        predicted *= np.linalg.det(_cholesky_transform(np.asarray(old.mo_coefficients), s))
    assert abs(np.log(abs(predicted))) > MO_ORTHONORMALITY_WARNING_THRESHOLD
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval"), "strict")
    positions = np.asarray(_reference(fixture, "up").atom_positions)
    log_ratios, relative_ratios = [], []
    for r_up, r_dn in _configurations(geminal_new, positions, 20261002, 24):
        ratio = complex(compute_det_geminal_all_elements(geminal_new, r_up, r_dn)) / complex(
            compute_det_geminal_all_elements(geminal_old, r_up, r_dn)
        )
        log_ratios.append(np.log(abs(ratio)))
        relative_ratios.append(ratio / predicted)
    np.testing.assert_allclose(log_ratios, np.log(abs(predicted)), atol=atol, rtol=rtol, equal_nan=False)
    # Modulus and phase together: the complex ratio over the prediction is one.
    np.testing.assert_allclose(
        relative_ratios, np.ones(len(relative_ratios), dtype=complex), atol=atol, rtol=rtol, equal_nan=False
    )


@pytest.mark.light
@pytest.mark.parametrize("fixture", ["o-lsda-gamma", "o-lsda-k025"])
def test_lsda_spins_share_one_overlap_and_are_orthonormalized_independently(fixture):
    """LSDA: one ``S_AO(k)`` per k for both spins; each spin block is its own Cholesky transform.

    During the conversion the converter evaluates ``compute_overlap_matrix``
    exactly once (shared by the up and dn blocks at the same k); that shared
    overlap equals the overlap of the separately built dn leaf, and each
    converted block equals the independent NumPy ``conj(L)^{-1} C`` of its own
    raw block (so the dn block is not the up transform applied to dn).
    """
    calls = []
    original = qe_to_jqmc.compute_overlap_matrix

    def counting(aos_data):
        calls.append(aos_data)
        return original(aos_data)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qe_to_jqmc, "compute_overlap_matrix", counting)
        _, up_new, dn_new = _convert(fixture, "pgto")
    assert len(calls) == 1
    _, up_old, dn_old = _convert(fixture, "pgto", orthonormalize=False)
    s_up = np.asarray(compute_overlap_matrix(up_old.aos_data))
    s_dn = np.asarray(compute_overlap_matrix(dn_old.aos_data))
    np.testing.assert_array_equal(s_dn, s_up)
    for new, old in ((up_new, up_old), (dn_new, dn_old)):
        c_old = np.asarray(old.mo_coefficients)
        expected = _cholesky_transform(c_old, s_up) @ c_old
        np.testing.assert_allclose(
            np.asarray(new.mo_coefficients),
            expected,
            atol=ATOL_PW_PROJECTION_FP64,
            rtol=RTOL_PW_PROJECTION_FP64,
            equal_nan=False,
        )
    # The up transform applied to the dn block is not the dn result (each spin has its own L).
    cross = _cholesky_transform(np.asarray(up_old.mo_coefficients), s_up)
    assert cross.shape[0] == np.asarray(dn_old.mo_coefficients).shape[0]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            cross @ np.asarray(dn_old.mo_coefficients),
            np.asarray(dn_new.mo_coefficients),
            atol=ATOL_PW_PROJECTION_FP64,
            rtol=RTOL_PW_PROJECTION_FP64,
            equal_nan=False,
        )


# ---------------------------------------------------------------------------
# 8. Mixed and unprojected kinds are never orthonormalized
# ---------------------------------------------------------------------------
@pytest.mark.light
@pytest.mark.parametrize("kind", ["pw", "bspline", "pgto+pw", "pgto+bspline"])
def test_orthonormalize_flag_does_not_touch_other_kinds(kind):
    """``orthonormalize`` is a no-op for pw, bspline, pgto+pw and pgto+bspline.

    The Li k = 1/4 all-band projections converted with the flag on (default)
    and off give bit-identical MO coefficient blocks in both spin channels,
    and the full block equals the plain packaging of the projection (no
    Gamma phase step at this twist): the PW block is the projection's
    ``pw_coefficients``, the spline block the identity, and the pGTO block the
    projection coefficients divided by an independent Gamma-function
    normalization, within the fp64 projection pair tolerance. In particular
    no part of a mixed basis is orthonormalized or rescaled apart from its PW /
    spline partner.
    """
    geminal_on, up_on, _ = _convert("li-k025", kind)
    geminal_off, up_off, _ = _convert("li-k025", kind, orthonormalize=False)
    np.testing.assert_array_equal(np.asarray(up_on.mo_coefficients), np.asarray(up_off.mo_coefficients))
    for a, b in (
        (geminal_on.orb_data_up_spin, geminal_off.orb_data_up_spin),
        (geminal_on.orb_data_dn_spin, geminal_off.orb_data_dn_spin),
    ):
        np.testing.assert_array_equal(np.asarray(a.mo_coefficients), np.asarray(b.mo_coefficients))
    proj = _projection("li-k025", kind)
    blocks = []
    if kind.startswith("pgto"):
        normalization = np.array([_independent_normalization(p) for p in proj.primitives])
        blocks.append(np.asarray(proj.coefficients) / normalization[None, :])
    if kind.endswith("pw"):
        blocks.append(np.asarray(proj.pw_coefficients))
    if kind.endswith("bspline"):
        blocks.append(np.eye(proj.band_indices.size))
    np.testing.assert_allclose(
        np.asarray(up_on.mo_coefficients),
        np.concatenate(blocks, axis=1),
        atol=ATOL_PW_PROJECTION_FP64,
        rtol=RTOL_PW_PROJECTION_FP64,
        equal_nan=False,
    )


@pytest.mark.light
def test_twist_average_writer_orthonormalizes_each_pgto_twist(tmp_path):
    """The combined twist writer threads ``orthonormalize`` to every per-twist pGTO block.

    The committed O LSDA k = 1/4 save is written by
    ``convert_qe_to_twist_average_input(kind="pgto")`` with the default and
    with ``orthonormalize=False`` and reloaded with ``load_twist``. With the
    default, both stored spin channels (4 up and 2 dn occupied MOs) are
    orthonormal in the overlap of their own loaded leaf; with the flag off the
    same comparison fails. Full precision uses the fp64 projection pair
    tolerance. In mixed precision the HDF5 round trip stores the pGTO
    exponents at ao_eval width (relative change 3e-8, overlap change 2.5e-8),
    so the comparison uses the strict ao_eval tolerance of that mode.
    """
    from jqmc._twist_average import load_twist

    if is_mixed_precision_enabled():
        atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    else:
        atol, rtol = ATOL_PW_PROJECTION_FP64, RTOL_PW_PROJECTION_FP64

    save, upf, _ = _SAVES["o-lsda-k025"]
    for orthonormalize in (True, False):
        path = tmp_path / f"twists_{orthonormalize}.h5"
        qe_to_jqmc.convert_qe_to_twist_average_input(
            str(save), str(path), upf_paths={"O": str(save / upf)}, kind="pgto", orthonormalize=orthonormalize
        )
        geminal = load_twist(str(path), 0).wavefunction_data.geminal_data
        for channel in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            gram = _gram(channel)
            if orthonormalize:
                np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=atol, rtol=rtol, equal_nan=False)
            else:
                with pytest.raises(AssertionError):
                    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.light
@pytest.mark.parametrize("flag", ["default", "--orthonormalize", "--no-orthonormalize"])
def test_cli_orthonormalize_option_reaches_converter(monkeypatch, flag):
    """``jqmc-tool qe`` exposes ``--orthonormalize/--no-orthonormalize`` (default on) to run_qe2jqmc."""
    from click.testing import CliRunner

    from jqmc.jqmc_tool import typer_click_qe

    captured = {}

    def fake_run_qe2jqmc(*, qe_params, jastrow_params, hamiltonian_h5):
        captured["qe_params"] = qe_params

    monkeypatch.setattr("jqmc.jqmc_qe.run_qe2jqmc", fake_run_qe2jqmc)
    args = [str(_SAVES["li-gamma"][0]), "-k", "pgto", "-o", "unused.h5"] + ([] if flag == "default" else [flag])
    result = CliRunner().invoke(typer_click_qe, args)
    assert result.exit_code == 0, result.output
    assert captured["qe_params"]["orthonormalize"] is (flag != "--no-orthonormalize")


# ---------------------------------------------------------------------------
# Unit-cell normalization of every kind
# ---------------------------------------------------------------------------
def _expected_cell_norm(fixture, proj, mos):
    """Reference ``integral_Omega |psi_n|^2`` of the converted bands, in the production overlap dtype.

    The projection's own norm is ``1 - r_n`` (fp64 QE data). The converted
    band's plane-wave block contributes ``V sum_G |d_{n,G}|^2`` of it
    (``d = pw_coefficients``); its cell metric is the analytic plane-wave
    overlap ``V delta`` (``V/2`` for the cos/sin pairs), which production
    stores in the ``ao_eval`` dtype for the real Gamma plane-wave leaf
    (the complex leaf's analytic overlap is complex128 in both modes). The
    reference evaluates that metric in the same dtype: the plane-wave block
    norm is scaled by ``fl(V) / V`` with ``fl`` the cast to the ``ao_eval``
    dtype. In full precision the factor is exactly one and the reference is
    exactly ``1 - r_n``.
    """
    leaves = mos.aos_data.children if isinstance(mos.aos_data, AOs_composite_data) else (mos.aos_data,)
    expected = 1.0 - np.asarray(proj.residuals)
    if not any(isinstance(leaf, AOs_plane_wave_data) for leaf in leaves):
        return expected
    volume = float(_reference(fixture, "up" if _SAVES[fixture][2] else None).cell.volume)
    metric_ratio = float(np.asarray(volume, dtype=get_dtype_np("ao_eval"))) / volume
    pw = np.asarray(proj.pw_coefficients)
    pw_block_norm = volume * np.real(np.einsum("ng,ng->n", pw.conj(), pw))
    return expected + (metric_ratio - 1.0) * pw_block_norm


@pytest.mark.parametrize(
    "fixture,kind",
    [
        pytest.param("li-gamma", "pw", marks=pytest.mark.light),
        pytest.param("li-k025", "pw", marks=pytest.mark.light),
        pytest.param("li-gamma", "bspline", marks=pytest.mark.light),
        pytest.param("li-gamma", "pgto+pw", marks=pytest.mark.light),
        pytest.param("li-k025", "pgto+pw", marks=pytest.mark.light),
        pytest.param("li-gamma", "pgto+bspline", marks=pytest.mark.heavy),
        pytest.param("li-k025", "pgto+bspline", marks=pytest.mark.light),
    ],
)
def test_converted_mos_are_unit_cell_normalized(fixture, kind):
    """``integral_Omega |psi_n|^2 = 1 - r_n`` for the PW, B-spline and mixed conversions.

    The diagonal of ``conj(C) S C^T`` (``S = compute_overlap_matrix`` of the
    converted leaf: analytic for plane waves, the cell quadrature for B-spline
    and composite leaves) is compared with ``1 - r_n`` (``r_n`` the
    projection's own residual) for every converted band of both spin channels
    of the occupied-band conversion. Without unit-cell normalization the same
    diagonal would be ``Omega (1 - r_n)`` (Omega = 292 bohr^3 here). The plane-wave overlap is
    analytic: full precision uses the fp64 projection pair tolerance, mixed the
    strict ao_eval tolerance (the real Gamma PW overlap is stored in the
    ao_eval dtype). The reference evaluates the plane-wave block's cell
    metric in that same dtype (``_expected_cell_norm``), so mixed is judged
    within its own precision. The quadrature-based leaves use the active-mode
    medium ao/mo zone tolerance.
    """
    lsda = _SAVES[fixture][2]
    proj = _projection(fixture, kind, "up" if lsda else None, True)
    geminal, up, _ = _convert(fixture, kind, occ_only=True)
    if kind == "pw" and not is_mixed_precision_enabled():
        atol, rtol = ATOL_PW_PROJECTION_FP64, RTOL_PW_PROJECTION_FP64
    elif kind == "pw":
        atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    else:
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval"), "medium")
    expected = _expected_cell_norm(fixture, proj, up)
    np.testing.assert_allclose(np.real(np.diag(_gram(up))), expected, atol=atol, rtol=rtol, equal_nan=False)
    for channel in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        diagonal = np.real(np.diag(_gram(channel)))
        np.testing.assert_allclose(diagonal, expected[: diagonal.size], atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.light
@pytest.mark.parametrize("fixture", ["li-gamma", "li-k025"])
def test_unorthonormalized_pgto_projection_has_cell_normalized_scale(fixture):
    """With ``orthonormalize=False`` the pGTO MOs keep the projection norm, not ``Omega`` times it.

    The diagonal of ``conj(C) S_E C^T`` -- ``S_E = G^H G`` on the QE wavefunction
    G-set, the metric of the least-squares projection -- equals ``1 - r_n`` for
    every band within the fp64 projection pair tolerance (a ``sqrt(Omega) c``
    right-hand side gives ``Omega (1 - r_n)``).
    """
    ref = _reference(fixture)
    proj = _projection(fixture, "pgto")
    design = build_gto_pw_matrix(ref, list(proj.primitives))
    c = np.asarray(proj.coefficients)
    diagonal = np.real(np.einsum("ni,ij,nj->n", c.conj(), design.conj().T @ design, c))
    np.testing.assert_allclose(
        diagonal, 1.0 - np.asarray(proj.residuals), atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64, equal_nan=False
    )
